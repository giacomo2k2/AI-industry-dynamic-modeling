import torch
from torch import nn
import math
import torch.autograd as autograd


# =====================================================
# ==========       Device Setup          ==============
# =====================================================

def device_setup():
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if my_device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_reserved(my_device)/1e9} GB")
    
    return my_device

# =====================================================
# ==========Recurrent Physics layer Network============
# =====================================================

# Dynamical Layer representing the double pendulum dynamics

class DynLayer(nn.Module):
    
    def __init__(
            self,
            sin_cos_transform,
            X_feature_positions_map,
            Y_feature_position_map,
            normalization,
            normalization_map,
            b1_ref: float = 1.5, fixed_b1: float = None,
            b2_ref: float = 0.8, fixed_b2: float = None, 
            l1_ref: float = 0.1, fixed_l1: float = None, 
            l2_ref: float = 5.0, fixed_l2: float = None, 
            m1_ref: float = 2.2, fixed_m1: float = None, 
            m2_ref: float = 0.5, fixed_m2: float = None
        ):
        
        super().__init__()

        self.sin_cos_transform = sin_cos_transform
        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.normalization = normalization
        self.normalization_map = normalization_map

        self.b1_ref, self.b2_ref = b1_ref, b2_ref
        self.l1_ref, self.l2_ref = l1_ref, l2_ref
        self.m1_ref, self.m2_ref = m1_ref, m2_ref
        
        def init_param(fixed_val, ref_val):
            if fixed_val is None:
                return nn.Parameter( torch.randn(1) * 0.1)
            else:
                val = math.log(fixed_val / ref_val)
                return torch.tensor([val], dtype=torch.float32)

        # Funzione helper per assegnare correttamente Parametri o Buffer
        def setup_variable(name, fixed_val, ref_val):
            val = init_param(fixed_val, ref_val)
            if fixed_val is None:
                # Se è None, è un nn.Parameter: lo assegnamo all'attributo
                setattr(self, name, val)
            else:
                # Se è fisso, lo registriamo come buffer.
                # NOTA: Non assegniamo il risultato a self.name! 
                # PyTorch renderà il buffer accessibile via self.name automaticamente.
                self.register_buffer(name, val)

        # Ora inizializza le variabili usando l'helper
        setup_variable("logb1", fixed_b1, b1_ref)
        setup_variable("logb2", fixed_b2, b2_ref)
        setup_variable("logl1", fixed_l1, l1_ref)
        setup_variable("logl2", fixed_l2, l2_ref)
        setup_variable("logm1", fixed_m1, m1_ref)
        setup_variable("logm2", fixed_m2, m2_ref)

    # Helper getters
    def get_b1(self): return torch.exp(self.logb1) * self.b1_ref   
    def get_b2(self): return torch.exp(self.logb2) * self.b2_ref   
    def get_l1(self): return torch.exp(self.logl1) * self.l1_ref   
    def get_l2(self): return torch.exp(self.logl2) * self.l2_ref   
    def get_m1(self): return torch.exp(self.logm1) * self.m1_ref   
    def get_m2(self): return torch.exp(self.logm2) * self.m2_ref

    def denormalize(self, states, features_position_map):
        # states: (Batch, state_dim)
        device = states.device
        denorm_states = torch.zeros_like(states, device=device)

        for feature, pos in features_position_map.items():
            if pos < states.shape[1]:
                if feature in self.normalization_map:
                    mean = self.normalization_map[feature]['mean']
                    std = self.normalization_map[feature]['std']
                    denorm_states[:, pos] = states[:, pos] * std + mean
                else:
                    denorm_states[:, pos] = states[:, pos]

        return denorm_states

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        inputs: 
            if self.sin_cos_transform:
                [Batch, 6] -> [q1_sin, q1_cos, q1_dot, q2_sin, q2_cos, q2_dot]
            else:
                [Batch, 4] -> [q1, q1_dot, q2, q2_dot]
        """

        # If the input is normalized, denormalization required to compute dynamics correctly
        if self.normalization:
            inputs = self.denormalize(inputs, self.X_feature_positions_map)

        inputs = inputs.requires_grad_(True) 
        
        # Recupero parametri (scalari broadcastati automaticamente)
        m1, m2 = self.get_m1(), self.get_m2()
        l1, l2 = self.get_l1(), self.get_l2()
        b1, b2 = self.get_b1(), self.get_b2()
        g = 9.81
        
        # Unpack input lungo la dimensione 1. 
        # q1 sarà di shape [Batch]
        if self.sin_cos_transform:
            q1_sin = inputs[:, self.X_feature_positions_map['q1_sin']]
            q1_cos = inputs[:, self.X_feature_positions_map['q1_cos']]
            q1 = torch.atan2(q1_sin, q1_cos)

            q1_dot = inputs[:, self.X_feature_positions_map['q1_dot']]

            q2_sin = inputs[:, self.X_feature_positions_map['q2_sin']]
            q2_cos = inputs[:, self.X_feature_positions_map['q2_cos']]
            q2 = torch.atan2(q2_sin, q2_cos)

            q2_dot = inputs[:, self.X_feature_positions_map['q2_dot']]
            
        else:
            q1 = inputs[:, self.X_feature_positions_map['q1']]
            q1_dot = inputs[:, self.X_feature_positions_map['q1_dot']]
            q2 = inputs[:, self.X_feature_positions_map['q2']]
            q2_dot = inputs[:, self.X_feature_positions_map['q2_dot']]

        # Termini trigonometrici [Batch]
        c2 = torch.cos(q2)
        s2 = torch.sin(q2)
        s1 = torch.sin(q1)
        s12 = torch.sin(q1 + q2)

        # === 1. Inertia Matrix M(q) [Batch, 2, 2] ===
        # Costruiamo i componenti uno per uno
        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2**2 + m2 * l1 * l2 * c2
        M21 = M12
        M22 = m2 * l2**2 * torch.ones_like(c2) # Assicura che sia un tensore [Batch]

        # Stack per creare matrice (Batch, 2, 2)
        # Row 1: [M11, M12], Row 2: [M21, M22]
        M = torch.stack([
            torch.stack([M11, M12], dim=1),
            torch.stack([M21, M22], dim=1)
        ], dim=1) 

        # === 2. Coriolis Vector c(q, q_dot) [Batch, 2, 1] ===
        c1 = -m2 * l1 * l2 * s2 * (2 * q1_dot + q2_dot) * q2_dot
        c2_elem = m2 * l1 * l2 * s2 * q1_dot**2
        
        c = torch.stack([c1, c2_elem], dim=1).unsqueeze(-1) 

        # === 3. Damping Matrix D [Batch, 2, 2] ===
        # D deve essere [Batch, 2, 2]. b1 e b2 sono scalari (o [1]), li espandiamo.
        batch_size = inputs.shape[0]
        zeros = torch.zeros(batch_size, device=inputs.device)
        ones = torch.ones(batch_size, device=inputs.device)
        
        # Espandi b1 e b2 alla dimensione del batch
        B1 = b1 * ones
        B2 = b2 * ones

        D = torch.stack([
            torch.stack([B1, zeros], dim=1),
            torch.stack([zeros, B2], dim=1)
        ], dim=1)

        # === 4. Gravity Vector G [Batch, 2, 1] ===
        G1 = (m1 + m2) * g * l1 * s1 + m2 * g * l2 * s12
        G2 = m2 * g * l2 * s12
        G = torch.stack([G1, G2], dim=1).unsqueeze(-1)

        # === 5. q_dot vector [Batch, 2, 1] ===
        q_dot_vec = torch.stack([q1_dot, q2_dot], dim=1).unsqueeze(-1)

        # === Solve M @ q_ddot = rhs ===
        # rhs shape: [Batch, 2, 1]
        rhs = -c - torch.bmm(D, q_dot_vec) - G  
        
        det_M = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        det_M = det_M.view(-1, 1, 1) # Reshape per broadcasting [Batch, 1, 1]

        inv_M = torch.stack([
            torch.stack([ M[:, 1, 1], -M[:, 0, 1]], dim=1),
            torch.stack([-M[:, 1, 0],  M[:, 0, 0]], dim=1)
        ], dim=1) / (det_M + 1e-6) # 1e-6 per evitare divisioni per zero

        # Risolvi sistema lineare per tutto il batch in parallelo
        q_ddot_batched = torch.matmul(inv_M, rhs)
        
        # Remove last dim -> [Batch, 2]
        q_ddot = q_ddot_batched.squeeze(-1) 


        # Concatena [q_dot, q_ddot] -> [Batch, 4]
        # q_dot_vec spremuto è [Batch, 2]
        
        
        # I OUTPUT ORDINAMENTO: [q1_dot, q1_ddot, q2_dot, q2_ddot]
        next_states = torch.stack([q_dot_vec.squeeze(-1)[:,0], q_ddot[:, 0], q_dot_vec.squeeze(-1)[:,1], q_ddot[:, 1]], dim=1)
        
        return next_states
    
class DynNN(nn.Module):
    '''
    same as DynLayer but for UDE but with NN instead of physical parameters
    there are no physical parameters, but a NN that learns the dynamics
    '''
    def __init__(
            self,
            X_feature_positions_map,
            Y_feature_position_map,
            Y_feature_DynLayer_positions_map,
            sin_cos_transform,
            normalization,
            normalization_map,
            input_dim, 
            output_dim, 
            hidden_dim=256, 
            activation='softplus', 
            residual=False
        ):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sin_cos_transform = sin_cos_transform

        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map
        self.normalization = normalization
        self.normalization_map = normalization_map

        self.residual = (residual and input_dim == output_dim)

        if activation == 'tanh':
            self.activation_func = nn.Tanh()
        elif activation == 'softplus':
            self.activation_func = nn.Softplus()
        elif activation == 'silu':
            self.activation_func = nn.SiLU()
        elif activation == 'relu':
            self.activation_func = nn.ReLU()
        elif activation == 'gelu':
            self.activation_func = nn.GELU()
        else:
            print(f"\n\t[ERROR] '{activation}' not recognized as a valid input for activation function \n")
            exit()
        
        # --- LAYER & PROFONDITÀ ---
        # 3 Layer Nascosti sono il "Sweet Spot" per la fisica.
        # Hidden Dim: 256 o 512 è sufficiente.
        
        # --- ACTIVATION FUNCTION ---
        # Usiamo Tanh o Softplus. 
        # MAI usare ReLU per la fisica HNN.
        # Motivo: ReLU ha derivata seconda zero ovunque (o indefinita). 
        # La fisica richiede derivate seconde continue (liscie).
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation_func, 
            nn.Linear(hidden_dim, hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, output_dim)  # Output: Scalare (L'Hamiltoniana / Energia)
        )

    def denormalize(self, states, features_position_map):
        # states: (Batch, state_dim)
        device = states.device
        denorm_states = torch.zeros_like(states, device=device)

        for feature, pos in features_position_map.items():
            if pos < states.shape[1]:
                if feature in self.normalization_map:
                    mean = self.normalization_map[feature]['mean']
                    std = self.normalization_map[feature]['std']
                    denorm_states[:, pos] = states[:, pos] * std + mean
                else:
                    denorm_states[:, pos] = states[:, pos]

        return denorm_states

    def forward(self, x):

        # Importante: abilitiamo il gradiente sull'input per calcolare le derivate parziali
        x = x.requires_grad_(True)
        
        if self.residual:
            outputs = x + self.net(x)
        else:
            outputs = self.net(x)

        if self.output_dim == 2: #TODO checking output dim
            # outputs: -> [q1_dot, q2_dot]
            # stack also q1, q2 from input
            #if self.sin_cos_transform:
            #    q1_sin = x[:, self.X_feature_positions_map['q1_sin']]
            #    q1_cos = x[:, self.X_feature_positions_map['q1_cos']]
            #    q1 = torch.atan2(q1_sin, q1_cos)

            #    q2_sin = x[:, self.X_feature_positions_map['q2_sin']]
            #    q2_cos = x[:, self.X_feature_positions_map['q2_cos']]
            #    q2 = torch.atan2(q2_sin, q2_cos)
            #else:
            #    q1 = x[:, self.X_feature_positions_map['q1']]
            #    q2 = x[:, self.X_feature_positions_map['q2']]
            #outputs = torch.stack([q1, outputs[:, 0], q2, outputs[:, 1]], dim=1)
            q1_dot = x[:, self.X_feature_positions_map['q1_dot']]
            q2_dot = x[:, self.X_feature_positions_map['q2_dot']]

            # I OUTPUT ORDINAMENTO: [q1_dot, q1_ddot, q2_dot, q2_ddot]
            outputs = torch.stack([q1_dot, outputs[:, 0], q2_dot,  outputs[:, 1]], dim=1)


        if self.normalization:
            outputs = self.denormalize(outputs, self.Y_feature_DynLayer_positions_map)

        return outputs
    
class DynUDE(nn.Module):
    '''
    same as DynLayer but for UDE but with NN instead of physical parameters
    there are no physical parameters, but a NN that learns the dynamics
    '''
    def __init__(
            self,
            sin_cos_transform,
            X_feature_positions_map,
            Y_feature_position_map,
            Y_feature_DynLayer_positions_map,
            normalization,
            normalization_map,
            input_dim=4, 
            hidden_dim=256, 
            output_dim=10, 
            activation='softplus'
        ):

        super().__init__()

        self.sin_cos_transform = sin_cos_transform
        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map
        self.normalization = normalization
        self.normalization_map = normalization_map
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)  # Output: Scalare (L'Hamiltoniana / Energia)
        )

    def forward(self, x):
        """
        x: 
            if self.sin_cos_transform:
                [Batch, 6] -> [q1_sin, q1_cos, q1_dot, q2_sin, q2_cos, q2_dot]
            else:
                [Batch, 4] -> [q1, q1_dot, q2, q2_dot]
        
        the forward pass predicts the next states given the current states and inputs
        outputs: [Batch, 10] -> [M11, M12, M21, M22, c1, c2, D11, D22, G1, G2]

        then dynamics are computed
        returns: [Batch, 4] -> [q1_dot, q1_ddot, q2_dot, q2_ddot]
        """

        # Importante: abilitiamo il gradiente sull'input per calcolare le derivate parziali
        x = x.requires_grad_(True)

        if self.normalization:
            x = self.denormalize(x, self.X_feature_positions_map)
        
        # 1. La rete predice l'Energia Totale (H) in quello stato
        outputs = self.net(x)

        if self.normalization:
            raise NotImplementedError("Denormalization not implemented for DynUDE_Informed yet.")

        M11, M12 = outputs[:, 0], outputs[:, 1]
        M21, M22 = outputs[:, 2], outputs[:, 3]
        c1, c2   = outputs[:, 4], outputs[:, 5]
        D11, D22 = outputs[:, 6], outputs[:, 7]
        G1, G2   = outputs[:, 8], outputs[:, 9]

        # --- 3. Costruzione Matrici Stimate ---
        # Matrice di Inerzia M
        M = torch.stack([
            torch.stack([M11, M12], dim=1),
            torch.stack([M21, M22], dim=1)
        ], dim=1)  # Shape: [Batch, 2, 2]
        
        # Vettore Coriolis/Centrifuga C (nota: nel tuo codice dynamics era 'c')
        C = torch.stack([c1, c2], dim=1).unsqueeze(-1)  # Shape: [Batch, 2, 1]
        
        # Matrice Damping D
        D = torch.stack([
            torch.stack([D11, torch.zeros_like(D11)], dim=1),
            torch.stack([torch.zeros_like(D22), D22], dim=1)
        ], dim=1)  # Shape: [Batch, 2, 2]
        
        # Vettore Gravità G
        G = torch.stack([G1, G2], dim=1).unsqueeze(-1)  # Shape: [Batch, 2, 1]

        # --- 4. Calcolo della Dinamica (Forward Dynamics) ---
        # Estraiamo le velocità dallo stato attuale [q1, q1_dot, q2, q2_dot]
        q1_dot = x[:, self.X_feature_positions_map['q1_dot']]
        q2_dot = x[:, self.X_feature_positions_map['q2_dot']]
        
        q_dot = torch.stack([q1_dot, q2_dot], dim=1).unsqueeze(-1)  # Shape: [Batch, 2, 1]
        
        # Equazione del moto: M * q_ddot + C + D*q_dot + G = u
        # Quindi: M * q_ddot = u - C - D*q_dot - G
        # Termine noto b:
        b = - C - torch.bmm(D, q_dot) - G  # Shape: [Batch, 2, 1]
        
        # Risolviamo per le accelerazioni (q_ddot)
        # Gestiamo eventuali matrici singolari con un try-except o pseudo-inversa
        try:
            q_ddot_est = torch.linalg.solve(M, b)  # Shape: [Batch, 2, 1]
        except RuntimeError:
            q_ddot_est = torch.matmul(torch.linalg.pinv(M), b)  # Shape: [Batch, 2, 1]

        q_ddot = q_ddot_est.squeeze(-1)  # Shape: [Batch, 2]

        # --- 5. Costruzione del Vettore di Output ---
        # Concatenazione di [q1_dot, q1_ddot, q2_dot, q2_ddot]
        #print("q_dot shape:", q_dot.squeeze(-1).shape)
        #print("q_ddot shape:", q_ddot.shape)

        derivatives = torch.stack([q_dot.squeeze(-1)[:, 0], q_ddot[:, 0], q_dot.squeeze(-1)[:, 1], q_ddot[:, 1]], dim=1)  # Shape: [Batch, 4]
        
        return derivatives
    
class DynUDE_Informed(nn.Module):
    def __init__(
            self,
            sin_cos_transform,
            X_feature_positions_map,
            Y_feature_position_map,
            Y_feature_DynLayer_positions_map,
            normalization,
            normalization_map,
            input_dim=4, 
            hidden_dim=256, 
            output_dim=9, 
            activation='softplus'
        ):

        super().__init__()

        self.sin_cos_transform = sin_cos_transform
        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map
        self.normalization = normalization
        self.normalization_map = normalization_map 
        
        # M (Simmetrica): serve L11, L21, L22 -> 3 parametri
        # C (Coriolis): c1, c2 -> 2 parametri
        # D (Damping diagonale): d1, d2 -> 2 parametri
        # G (Gravity): g1, g2 -> 2 parametri
        # TOTALE: 9 parametri

        self.softplus = nn.Softplus() # Per garantire positività dove necessario

        if activation == 'tanh':
            self.activation_func = nn.Tanh()
        elif activation == 'softplus':
            self.activation_func = nn.Softplus()
        elif activation == 'silu':
            self.activation_func = nn.SiLU()
        elif activation == 'relu':
            self.activation_func = nn.ReLU()
        elif activation == 'gelu':
            self.activation_func = nn.GELU()
        else:
            print(f"\n\t[ERROR] '{activation}' not recognized as a valid input for activation function \n")
            exit()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation_func,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [Batch, 4] -> [q1, q1_dot, q2, q2_dot]

        x = x.requires_grad_(True)
        
        # Predizione parametri grezzi
        raw_out = self.net(x)

        if self.normalization:
            raise NotImplementedError("Denormalization not implemented for DynUDE_Informed yet.")
        
        # --- 1. Costruzione Matrice M (Metodo Cholesky) ---
        # M deve essere definita positiva. Costruiamo L tale che M = L * L^T
        # L = [[l11, 0], [l21, l22]]
        
        l11 = self.softplus(raw_out[:, 0]) + 1e-4  # Deve essere > 0
        l21 = raw_out[:, 1]                        # Può essere negativo
        l22 = self.softplus(raw_out[:, 2]) + 1e-4  # Deve essere > 0
        
        # Costruiamo L batch-wise
        zeros = torch.zeros_like(l11)
        row1 = torch.stack([l11, zeros], dim=1)
        row2 = torch.stack([l21, l22], dim=1)
        L = torch.stack([row1, row2], dim=1) # [Batch, 2, 2]
        
        # M = L @ L.transpose
        M = torch.bmm(L, L.transpose(1, 2)) # [Batch, 2, 2]
        # Ora M è garantita simmetrica e invertibile! Niente più try-except.

        # --- 2. Damping D (Diagonale e Positivo) ---
        d1 = self.softplus(raw_out[:, 3])
        d2 = self.softplus(raw_out[:, 4])
        
        D = torch.stack([
            torch.stack([d1, zeros], dim=1),
            torch.stack([zeros, d2], dim=1)
        ], dim=1)

        # --- 3. Coriolis C ---
        c1 = raw_out[:, 5]
        c2 = raw_out[:, 6]
        C = torch.stack([c1, c2], dim=1).unsqueeze(-1)

        # --- 4. Gravity G ---
        g1 = raw_out[:, 7]
        g2 = raw_out[:, 8]
        G = torch.stack([g1, g2], dim=1).unsqueeze(-1)

        # --- 5. Risoluzione Dinamica ---
        q1_dot = x[:, self.X_feature_positions_map['q1_dot']]
        q2_dot = x[:, self.X_feature_positions_map['q2_dot']]
        q_dot = torch.stack([q1_dot, q2_dot], dim=1).unsqueeze(-1)
        
        # Termine noto: b = -C - D*q_dot - G
        b = -C - torch.bmm(D, q_dot) - G
        
        # Risolviamo M * q_ddot = b
        # Essendo M definita positiva per costruzione, solve è stabile.
        q_ddot_est = torch.linalg.solve(M, b) 
        
        q_ddot = q_ddot_est.squeeze(-1)
        # I OUTPUT ORDINAMENTO: [q1_dot, q1_ddot, q2_dot, q2_ddot]
        derivatives = torch.stack([q_dot.squeeze(-1)[:, 0], q_ddot[:, 0], q_dot.squeeze(-1)[:, 1], q_ddot[:, 1]], dim=1)  # Shape: [Batch, 4]
        
        return derivatives

# =====================================================
# ==========       Numerical Integrators      =========
# =====================================================

# Simple Euler integrator

class EulerModel(nn.Module):
    '''
    Docstring for EulerModel
    Simple Euler integrator for ODEs.
    Inputs:
        f: function representing the system dynamics, should take input of shape (Batch, state_dim) and return (Batch, state_dim)
        auxiliary_input: bool, if True, f will receive auxiliary inputs at each time step
    
    Outputs:
        forward method returns the integrated states over time of shape (Batch, time_steps, state_dim
    '''

    def __init__(
            self, 
            f,
            sin_cos_transform,
            X_feature_positions_map,
            Y_feature_position_map,
            Y_feature_DynLayer_positions_map,
            physical_state_positions_map,
            normalization,
            normalization_map,
            auxiliary_input=False
        ):
        super().__init__()

        self.f = f 
        self.sin_cos_transform = sin_cos_transform
        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map
        self.physical_state_positions_map = physical_state_positions_map
        self.normalization = normalization
        self.normalization_map = normalization_map

        self.auxiliary_input = auxiliary_input

    def integration(self, y0, T, aux=None):
        
        # y0 dim -> (Batch, state_dim)
        # T -> (Batch, time_steps, 1)
        
        # Nota: Qui assumiamo che T abbia step costanti o che T[i] sia lo stesso per tutto il batch
        # Se ogni campione ha tempi diversi, dobbiamo gestire dt vettorialmente.
        
        device = y0.device
        
        # Lista per salvare gli step. Iniziamo con lo stato iniziale
        res = [y0] 
        
        curr_y = y0 # (Batch, 4)

        # Loop temporale (questo non si può evitare con Euler, ma ora interno è vettorizzato)
        # Iteriamo su T.shape[1] (i time steps)
        num_steps = T.shape[1] - 1
        
        for i in range(num_steps):
            
            t_curr = T[:, i, 0]   # (Batch,)
            t_next = T[:, i+1, 0] # (Batch,)
            
            # Calcola derivata per TUTTO il batch
            if self.auxiliary_input:
                dy = self.f((curr_y, aux[:, i, :])) 
            else:
                dy = self.f(curr_y) # Restituisce (Batch, 4)

            # Calcola dt per ogni campione nel batch
            dt = (t_next - t_curr).unsqueeze(-1) # (Batch, 1) per broadcasting
            
            # Step di Eulero
            curr_y = curr_y + dt * dy

            if self.sin_cos_transform:
                # trasformiamo [q1,q1_dot,q2,q2_dot] in [sin(q1),cos(q1),q1_dot,sin(q2),cos(q2),q2_dot]
                q1 = curr_y[:, 0]
                q1_dot = curr_y[:, 1]
                q2 = curr_y[:, 2]
                q2_dot = curr_y[:, 3]

                q1_sin = torch.sin(q1)
                q1_cos = torch.cos(q1)
                q2_sin = torch.sin(q2)
                q2_cos = torch.cos(q2)

                curr_y = torch.zeros((curr_y.shape[0], 6), device=device)
                curr_y[:, self.X_feature_positions_map['q1_sin']] = q1_sin
                curr_y[:, self.X_feature_positions_map['q1_cos']] = q1_cos
                curr_y[:, self.X_feature_positions_map['q1_dot']] = q1_dot
                curr_y[:, self.X_feature_positions_map['q2_sin']] = q2_sin
                curr_y[:, self.X_feature_positions_map['q2_cos']] = q2_cos
                curr_y[:, self.X_feature_positions_map['q2_dot']] = q2_dot
            
            res.append(curr_y)

        # Stack lungo la dimensione temporale
        # res è una lista di [Batch, 4]. Stack -> [Time, Batch, 4]
        res = torch.stack(res, dim=1) # -> [Batch, Time, 4]
        return res

    def forward(self, inputs):
        # inputs: (Batch, time_steps, features)

        print("SHAPE INPUTS EULER:", inputs.shape)


        T_pos = self.X_feature_positions_map['time_step']
        
        if self.auxiliary_input:
            y0 = inputs[:, 0, :T_pos]      # Stato iniziale al tempo t=0 (Batch, 4)
            T =  inputs[:, :, T_pos:T_pos+1]     # Tempi (Batch, Time, 1)
            aux = inputs[:, :, T_pos:]
        else:
            y0 = inputs[:, 0, :T_pos]      # Stato iniziale (Batch, 4)
            T = inputs[:, :, T_pos:]       # Tempi (Batch, Time, 1)
            aux = None

        # Niente più ciclo sul batch size!
        res = self.integration(y0, T, aux)

        return res

# RK4 integrator
    
class RK4Model(nn.Module):
    '''
    Docstring for RK4Model
    Runge-Kutta 4th order integrator for ODEs.
    Inputs:
        f: function representing the system dynamics, should take input of shape (Batch, state_dim) and return (Batch, state_dim)
        auxiliary_input: bool, if True, f will receive auxiliary inputs at each time step
        additional_derivatives: bool, if True, forward returns also the derivatives at each time step
    Outputs:
        forward method returns the integrated states over time of shape (Batch, time_steps, state_dim)
        if additional_derivatives is True, also returns derivatives of shape (Batch, time_steps, state_dim)
    '''

    def __init__(
            self, 
            f,
            sin_cos_transform,
            normalization,
            normalization_map,
            X_feature_positions_map,
            Y_feature_position_map,
            Y_feature_DynLayer_positions_map,
            physical_state_positions_map,
            auxiliary_input=False, 
            additional_derivatives=False

        ):

        super().__init__()
        self.f = f
        self.sin_cos_transform = sin_cos_transform
        self.normalization = normalization
        self.normalization_map = normalization_map
        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map
        self.physical_state_positions_map = physical_state_positions_map
        self.auxiliary_input = auxiliary_input
        self.additional_derivatives = additional_derivatives

    def to_sin_cos(self, states, device):
        
        q1 = states[:, self.physical_state_positions_map['q1']]
        q1_dot = states[:,  self.physical_state_positions_map['q1_dot']]
        q2 = states[:, self.physical_state_positions_map['q2']]
        q2_dot = states[:,  self.physical_state_positions_map['q2_dot']]

        q1_sin = torch.sin(q1)
        q1_cos = torch.cos(q1)
        q2_sin = torch.sin(q2)
        q2_cos = torch.cos(q2)

        sin_cos_states = torch.zeros((states.shape[0], 6), device=device)
        sin_cos_states[:, self.X_feature_positions_map['q1_sin']] = q1_sin
        sin_cos_states[:, self.X_feature_positions_map['q1_cos']] = q1_cos
        sin_cos_states[:, self.X_feature_positions_map['q1_dot']] = q1_dot
        sin_cos_states[:, self.X_feature_positions_map['q2_sin']] = q2_sin
        sin_cos_states[:, self.X_feature_positions_map['q2_cos']] = q2_cos
        sin_cos_states[:, self.X_feature_positions_map['q2_dot']] = q2_dot

        return sin_cos_states
    
    def from_sin_cos(self, sin_cos_states, device):
        # trasformiamo [sin(q1),cos(q1),q1_dot,sin(q2),cos(q2),q2_dot] in [q1,q1_dot,q2,q2_dot]
        q1_sin = sin_cos_states[:, self.X_feature_positions_map['q1_sin']]
        q1_cos = sin_cos_states[:, self.X_feature_positions_map['q1_cos']]
        q1 = torch.atan2(q1_sin, q1_cos)

        q1_dot = sin_cos_states[:, self.X_feature_positions_map['q1_dot']]

        q2_sin = sin_cos_states[:, self.X_feature_positions_map['q2_sin']]
        q2_cos = sin_cos_states[:, self.X_feature_positions_map['q2_cos']]
        q2 = torch.atan2(q2_sin, q2_cos)

        q2_dot = sin_cos_states[:, self.X_feature_positions_map['q2_dot']]

        states = torch.zeros((sin_cos_states.shape[0], 4), device=device)        
        states[:, self.physical_state_positions_map['q1']] = q1
        states[:, self.physical_state_positions_map['q1_dot']] = q1_dot
        states[:, self.physical_state_positions_map['q2']] = q2
        states[:, self.physical_state_positions_map['q2_dot']] = q2_dot

        return states

    def normalize(self, denorm_states, features_position_map, device):
        # denorm_states: (Batch, state_dim)
        norm_states = torch.zeros_like(denorm_states, device=device)

        # create a list with unique features from both maps
        for feature, pos in features_position_map.items():
            if pos < denorm_states.shape[1]:
                if feature in self.normalization_map: 
                    mean = self.normalization_map[feature]['mean']
                    std = self.normalization_map[feature]['std']
                    norm_states[:, pos] = (denorm_states[:, pos] - mean) / std
                else:
                    norm_states[:, pos] = denorm_states[:, pos]

        return norm_states
    
    def denormalize(self, norm_states,features_position_map, device):
        # norm_states: (Batch, state_dim)
        denorm_states = torch.zeros_like(norm_states, device=device)

        for feature, pos in features_position_map.items():
            if pos < norm_states.shape[1]:
                if feature in self.normalization_map:
                    mean = self.normalization_map[feature]['mean']
                    std = self.normalization_map[feature]['std']
                    denorm_states[:, pos] = norm_states[:, pos] * std + mean
                else:
                    denorm_states[:, pos] = norm_states[:, pos]

        return denorm_states

    def from_training_to_physical(self, states, features_position_map, device):

        # states: (Batch, state_dim)
        physical_states = states

        if self.normalization:
            physical_states = self.denormalize(physical_states, features_position_map, device)
        if self.sin_cos_transform:
            physical_states = self.from_sin_cos(physical_states, device)

        return physical_states

    def from_physical_to_training(self, states, features_position_map, device):
        # states: (Batch, state_dim)
        training_states = states

        if self.normalization:
            training_states = self.normalize(training_states, features_position_map, device)
        if self.sin_cos_transform:
            training_states = self.to_sin_cos(training_states, device)

        return training_states

    def integration(self, y0, T, aux=None):
        # y0: (Batch, state_dim) -> (Batch, triaining_space_dim_without_aux)
        # T:  (Batch, time_steps, 1)
        # aux: (Batch, time_steps, aux_dim) or None
        
        device = y0.device
        
        # Lista per salvare gli step.
        res = [y0]
        der = [] 
        curr_y = y0 

        # Numero di step temporali
        num_steps = T.shape[1] - 1
        
        for i in range(num_steps):
            
            t_curr = T[:, i, 0]   # (Batch,)
            t_next = T[:, i+1, 0] # (Batch,)
            
            # Calcolo dt vettorizzato per ogni campione
            dt = (t_next - t_curr).unsqueeze(-1) # (Batch, 1)
            h = dt # alias per notazione RK4 classica
            
            # Auxiliary input at current time step (fixed for all RK4 sub-steps)
            current_aux = aux[:, i, :] if self.auxiliary_input else None
            
            # Converting current state to physical space (remove sin/cos and denormalize if needed)
            curr_y_physical = self.from_training_to_physical(curr_y, self.X_feature_positions_map, device)

            
            # ===========================================
            # ============= Integration RK4 =============
            # ===========================================
            

            # --------- 1)  k1 = f(tn, yn) ---------

            # prepare input for f
            if self.auxiliary_input:
                k1_input = torch.cat([curr_y, current_aux], dim=1)
            else:
                k1_input = curr_y
            # compute k1 (q_dot, q_dot_dot) 
            k1 = self.f(k1_input) # NB: we assume f returns derivatives in physical space

            # --------- 2) k2 = f(tn + h/2, yn + h*k1/2) ---------
            
            # compute y_k2
            y_k2_physical = curr_y_physical + h * 0.5 * k1
            # convert back to training space (add sin/cos and normalize if needed)
            y_k2 = self.from_physical_to_training(y_k2_physical, self.X_feature_positions_map, device)
            # prepare input for f
            if self.auxiliary_input:
                k2_input = torch.cat([y_k2, current_aux], dim=1)
            else:
                k2_input = y_k2
            # compute k2
            k2 = self.f(k2_input) # NB: we assume f returns derivatives in physical space

            # --------- 3) k3 = f(tn + h/2, yn + h*k2/2) ---------

            # compute y_k3
            y_k3_physical = curr_y_physical + h * 0.5 * k2
            # convert back to training space (add sin/cos and normalize if needed)
            y_k3 = self.from_physical_to_training(y_k3_physical, self.X_feature_positions_map, device)   
            # prepare input for f
            if self.auxiliary_input:
                k3_input = torch.cat([y_k3, current_aux], dim=1)
            else:
                k3_input = y_k3
            # compute k3
            k3 = self.f(k3_input) # NB: we assume f returns derivatives in physical space

            # --------- 4) k4 = f(tn + h, yn + h*k3) ---------

            # compute y_k4
            y_k4_physical = curr_y_physical + h * k3
            # convert back to training space (add sin/cos and normalize if needed)
            y_k4 = self.from_physical_to_training(y_k4_physical, self.X_feature_positions_map, device)
            # prepare input for f
            if self.auxiliary_input:
                k4_input = torch.cat([y_k4, current_aux], dim=1)
            else:
                k4_input = y_k4
            # compute k4
            k4 = self.f(k4_input) # NB: we assume f returns derivatives in physical space

            # --------- Update yn to yn+1 ---------
            # y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
            curr_y_physical = curr_y_physical + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            # convert back to training space (add sin/cos and normalize if needed)
            curr_y = self.from_physical_to_training(curr_y_physical, self.X_feature_positions_map, device)

            # --------- Append results ---------
            der.append(k1) # 1st derivative k1 for custom loss functions
            res.append(curr_y) # append integrated state in training space (with sin/cos and normalization if needed)

        # Append last 1st derivative k1 for custom loss functions
        der.append(self.f(curr_y))
        der = torch.stack(der, dim=1)

        # Stack lungo la dimensione temporale -> [Batch, Time, training_space_dim_without_aux]
        res = torch.stack(res, dim=1)
        
        return res, der

    def forward(self, inputs):
        # inputs: (Batch, time_steps, features)
        # La logica di slicing è identica a prima

        T_pos = self.X_feature_positions_map['time_step']
        
        if self.auxiliary_input:
            y0 = inputs[:, 0, :T_pos] # Shape (Batch, state_dim)
            T =  inputs[:, :, T_pos:T_pos+1]  
            aux = inputs[:, :, T_pos+1:]
        else:
            y0 = inputs[:, 0, :T_pos]      
            T = inputs[:, :, T_pos:]     
            aux = None

        #print(f"y0: {y0.shape}")
        #print(f"T: {T.shape}")

        res, der = self.integration(y0, T, aux)

        #print(f"res: {res.shape}")
        #print(f"der: {der.shape}")
        #exit()
        
        if self.additional_derivatives:
            return res, der
        else:
            return res
    
# =====================================================
# ==========            UDE                ============
# =====================================================

# Simple Deep Neural Network

class DeepNN(nn.Module):
    '''
    Docstring for DeepNN
    Simple Feedforward Deep Neural Network.
    Inputs:
        input_dim: int, dimension of input features
        output_dim: int, dimension of output features
        hidden_dims: list of int, dimensions of hidden layers
    Outputs:
        forward method returns output of shape (Batch, output_dim)
    '''
    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 hidden_dims: list[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim)) # TODO aggiungere skip connection
            layers.append(nn.ReLU())
            prev_dim = hdim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.network(x)
    


