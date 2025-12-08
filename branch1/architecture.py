import torch
from torch import nn
import math

# =====================================================
# ==========Recurrent Physics layer Network============
# =====================================================

class DynLayer(nn.Module):
    
    def __init__(self,
                 b1_ref: float = 0.5, fixed_b1: float = None,
                 b2_ref: float = 0.5, fixed_b2: float = None, 
                 l1_ref: float = 1.0, fixed_l1: float = None, 
                 l2_ref: float = 1.0, fixed_l2: float = None, 
                 m1_ref: float = 2.0, fixed_m1: float = None, 
                 m2_ref: float = 2.0, fixed_m2: float = None):
        
        super().__init__()

        self.b1_ref, self.b2_ref = b1_ref, b2_ref
        self.l1_ref, self.l2_ref = l1_ref, l2_ref
        self.m1_ref, self.m2_ref = m1_ref, m2_ref
        
        def init_param(fixed_val, ref_val):
            if fixed_val is None:
                return nn.Parameter( torch.randn(1) )
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [Batch, 4] -> [q1, q1_dot, q2, q2_dot]
        
        # Recupero parametri (scalari broadcastati automaticamente)
        m1, m2 = self.get_m1(), self.get_m2()
        l1, l2 = self.get_l1(), self.get_l2()
        b1, b2 = self.get_b1(), self.get_b2()
        g = 9.81
        
        # Unpack input lungo la dimensione 1. 
        # q1 sarà di shape [Batch]
        q1 = inputs[:, 0]
        q1_dot = inputs[:, 1]
        q2 = inputs[:, 2]
        q2_dot = inputs[:, 3]

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
        next_states = torch.cat([q_dot_vec.squeeze(-1), q_ddot], dim=1)
        
        return next_states
    
class EulerModel(nn.Module):
    def __init__(self, f, auxiliary_input=False):
        super().__init__()
        self.f = f 
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
            
            res.append(curr_y)

        # Stack lungo la dimensione temporale
        # res è una lista di [Batch, 4]. Stack -> [Time, Batch, 4]
        res = torch.stack(res, dim=1) # -> [Batch, Time, 4]
        return res

    def forward(self, inputs):
        # inputs: (Batch, time_steps, features)
        
        if self.auxiliary_input:
            y0 = inputs[:, 0, :4]      # Stato iniziale al tempo t=0 (Batch, 4)
            T =  inputs[:, :, 4:5]     # Tempi (Batch, Time, 1)
            aux = inputs[:, :, 5:]
        else:
            y0 = inputs[:, 0, :4]      # Stato iniziale (Batch, 4)
            T = inputs[:, :, 4:]       # Tempi (Batch, Time, 1)
            aux = None

        # Niente più ciclo sul batch size!
        res = self.integration(y0, T, aux)

        return res
    
    
class RK4Model(nn.Module):
    def __init__(self, f, auxiliary_input=False):
        super().__init__()
        self.f = f 
        self.auxiliary_input = auxiliary_input

    def integration(self, y0, T, aux=None):
        # y0: (Batch, state_dim) -> (Batch, 4)
        # T:  (Batch, time_steps, 1)
        
        device = y0.device
        
        # Lista per salvare gli step.
        res = [y0] 
        curr_y = y0 

        # Numero di step temporali
        num_steps = T.shape[1] - 1
        
        for i in range(num_steps):
            
            t_curr = T[:, i, 0]   # (Batch,)
            t_next = T[:, i+1, 0] # (Batch,)
            
            # Calcolo dt vettorizzato per ogni campione
            dt = (t_next - t_curr).unsqueeze(-1) # (Batch, 1)
            h = dt # alias per notazione RK4 classica
            
            # Gestione input ausiliari (es. forze esterne)
            # Nota: per rigore, k2 e k3 dovrebbero usare aux interpolato a t+h/2
            # Qui usiamo l'aux allo step 'i' per semplicità, compatibile con dataset discreti.
            current_aux = aux[:, i, :] if self.auxiliary_input else None
            
            # --- Passaggio RK4 ---
            
            # k1 = f(tn, yn)
            if self.auxiliary_input:
                k1 = self.f(torch.cat([curr_y, current_aux], dim=1))
            else:
                k1 = self.f(curr_y)
            
            # k2 = f(tn + h/2, yn + h*k1/2)
            y_k2 = curr_y + h * 0.5 * k1
            if self.auxiliary_input:
                k2 = self.f(torch.cat([y_k2, current_aux], dim=1))
            else:
                k2 = self.f(y_k2)
                
            # k3 = f(tn + h/2, yn + h*k2/2)
            y_k3 = curr_y + h * 0.5 * k2
            if self.auxiliary_input:
                k3 = self.f(torch.cat([y_k3, current_aux], dim=1))
            else:
                k3 = self.f(y_k3)
            
            # k4 = f(tn + h, yn + h*k3)
            y_k4 = curr_y + h * k3
            if self.auxiliary_input:
                k4 = self.f(torch.cat([y_k4, current_aux], dim=1))
            else:
                k4 = self.f(y_k4)
            
            # Aggiornamento stato
            # y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
            curr_y = curr_y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            res.append(curr_y)

        # Stack lungo la dimensione temporale -> [Batch, Time, 4]
        res = torch.stack(res, dim=1)
        return res

    def forward(self, inputs):
        # inputs: (Batch, time_steps, features)
        # La logica di slicing è identica a prima
        
        if self.auxiliary_input:
            y0 = inputs[:, 0, :4]      
            T =  inputs[:, :, 4:5]     
            aux = inputs[:, :, 5:]
        else:
            y0 = inputs[:, 0, :4]      
            T = inputs[:, :, 4:]       
            aux = None

        res = self.integration(y0, T, aux)
        return res
    
# =====================================================
# ==========            UDE                ============
# =====================================================

# inputs (q, qdot), ou

class DeepNN(nn.Module):

    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 hidden_dims: list[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

# ====================================================
# ==========          PINN Model             =========
# ====================================================

import torch
from pina.problem import TimeDependentProblem
from pina.domain import CartesianDomain
from pina import Condition, LabelTensor
from pina.equation import Equation
from pina.condition import InputTargetCondition

def grad(u, x):
    
    grads = torch.autograd.grad(
        outputs=u, 
        inputs=x, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True, 
        retain_graph=True,
        allow_unused=True 
    )[0]

    if grads is None:
        return torch.zeros_like(x)
    
    return grads


class NormalizedDoublePendulum(TimeDependentProblem):

    def __init__(self, 
                 train_data_input, 
                 train_data_output,
                 scale_t,          
                 scale_y,         
                 m1: float = 1.0,
                 m2: float = 1.0,
                 l1: float = 1.0,
                 l2: float = 1.0,
                 b1: float = 0.1,
                 b2: float = 0.1,
                 g: float = 9.81):
        
        self.scale_t = torch.tensor(scale_t, dtype=torch.float32)
        self.scale_y = torch.tensor(scale_y, dtype=torch.float32)

        self.input_data = None
        self.output_data = None
        
        if train_data_input is not None and train_data_output is not None:
            t_tensor = torch.as_tensor(train_data_input).float()
            y_tensor = torch.as_tensor(train_data_output).float()
            if t_tensor.ndim == 1: t_tensor = t_tensor.unsqueeze(1)
            
            self.input_data = LabelTensor(t_tensor, labels=['t'])
            self.output_data = LabelTensor(y_tensor, labels=['q1', 'q2', 'q1_dot', 'q2_dot'])

        # Parametri Fisici
        self.g = g
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.b1, self.b2 = b1, b2

        super().__init__()

    @property
    def conditions(self):
        conds = {}

        # --- Physics Loss (ODE) ---
        conds['physics'] = Condition(input=self.input_data, 
                                     equation=Equation(self.ode_system))

        # --- Data Loss ---
        conds['data'] = InputTargetCondition(input=self.input_data, 
                                                 target=self.output_data)
            
        # --- Initial Condition ---
        # enforce the penalty constrain that y^(t=0) = y0
        t0_mask = (self.input_data['t'] == 0).squeeze()
        if t0_mask.any():
            input_t0 = self.input_data[t0_mask]
            target_t0 = self.output_data[t0_mask]
            
            conds['initial'] = InputTargetCondition(input=input_t0,
                                                    target=target_t0)
        
        return conds

    @property
    def output_variables(self):
        return ['q1', 'q2', 'q1_dot', 'q2_dot']

    @property
    def temporal_domain(self):
        max_t_norm = max(self.input_data).item()
        return CartesianDomain({'t': [0, max_t_norm]})

    def ode_system(self, input_, output_):
        
        # 1. Variabili NORMALIZZATE (Dal mondo della rete neurale)
        t_norm = input_.extract(['t'])
        t_norm.requires_grad_(True)
        
        q1_norm = output_.extract(['q1'])
        q2_norm = output_.extract(['q2'])
        q1_dot_norm = output_.extract(['q1_dot'])
        q2_dot_norm = output_.extract(['q2_dot'])

        # 2. Calcolo Derivate NORMALIZZATE (dy_norm / dt_norm)
        # Queste sono le pendenze nel grafico normalizzato
        dt_q1_norm = grad(q1_norm, t_norm)
        dt_q2_norm = grad(q2_norm, t_norm)
        dt_q1_dot_norm = grad(q1_dot_norm, t_norm) 
        dt_q2_dot_norm = grad(q2_dot_norm, t_norm) 

        # 3. DENORMALIZZAZIONE (Chain Rule)
        # Vogliamo le grandezze fisiche per calcolare M, C, G
        
        # A. Variabili di stato Fisiche: y_phys = y_norm * scale_y
        q1_phys = q1_norm * self.scale_y[0]
        q2_phys = q2_norm * self.scale_y[1]
        # Nota: usiamo le velocità predette dalla rete (denormalizzate) per calcolare Coriolis/Damping
        q1_dot_phys_pred = q1_dot_norm * self.scale_y[2]
        q2_dot_phys_pred = q2_dot_norm * self.scale_y[3]

        # B. Derivate Fisiche: dy/dt = (dy_norm * scale_y) / (dt_norm * scale_t)
        # Quindi: dy_phys/dt_phys = dy_norm/dt_norm * (scale_y / scale_t)
        
        # Velocità (d_pos / d_time)
        vel_1_phys = dt_q1_norm * (self.scale_y[0] / self.scale_t)
        vel_2_phys = dt_q2_norm * (self.scale_y[1] / self.scale_t)
        
        # Accelerazioni (d_vel / d_time)
        acc_1_phys = dt_q1_dot_norm * (self.scale_y[2] / self.scale_t)
        acc_2_phys = dt_q2_dot_norm * (self.scale_y[3] / self.scale_t)

        # 4. FISICA (Calcolata usando SOLO valori FISICI)
        # Da qui in poi è uguale a prima, ma usiamo le variabili *_phys e *_pred
        
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        b1, b2 = self.b1, self.b2

        # --- Matrici (usiamo posizioni fisiche e velocità fisiche predette) ---
        # Cos e Sin lavorano sugli angoli fisici (radianti), non normalizzati!
        
        # Matrice Inerzia M
        M00 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * torch.cos(q2_phys)
        M01 = m2 * l2**2 + m2 * l1 * l2 * torch.cos(q2_phys)
        M10 = M01
        M11 = m2 * l2**2 
        
        det_M = M00 * M11 - M01 * M10

        # Coriolis C (usa velocità fisiche)
        term_c = m2 * l1 * l2 * torch.sin(q2_phys)
        C0 = -term_c * (2 * q1_dot_phys_pred + q2_dot_phys_pred) * q2_dot_phys_pred
        C1 = term_c * q1_dot_phys_pred**2

        # Damping D
        D0 = b1 * q1_dot_phys_pred
        D1 = b2 * q2_dot_phys_pred

        # Gravità G
        G0 = (m1 + m2) * g * l1 * torch.sin(q1_phys) + m2 * g * l2 * torch.sin(q1_phys + q2_phys)
        G1 = m2 * g * l2 * torch.sin(q1_phys + q2_phys)

        # 5. Dinamica Inversa
        b0 = -C0 - D0 - G0
        b1 = -C1 - D1 - G1

        acc_phys_target_1 = (M11 * b0 - M01 * b1) / det_M
        acc_phys_target_2 = (-M10 * b0 + M00 * b1) / det_M

        # 6. Residui (Confronto tra grandezze fisiche)
        
        # Residuo velocità: (Derivata Fisica della Posizione) - (Velocità Fisica Predetta)
        res_vel_1 = vel_1_phys - q1_dot_phys_pred
        res_vel_2 = vel_2_phys - q2_dot_phys_pred
        
        # Residuo accelerazione: (Derivata Fisica della Velocità) - (Accelerazione Fisica Target)
        res_acc_1 = acc_1_phys - acc_phys_target_1
        res_acc_2 = acc_2_phys - acc_phys_target_2

        return torch.hstack([res_vel_1, res_vel_2, res_acc_1, res_acc_2])

class DoppioPendolo(TimeDependentProblem):

    def __init__(self, 
                 train_data_input, 
                 train_data_output,
                 m1: float = 1.0,
                 m2: float = 1.0,
                 l1: float = 1.0,
                 l2: float = 1.0,
                 b1: float = 0.1,  # Smorzamento giunto 1
                 b2: float = 0.1,  # Smorzamento giunto 2
                 g: float = 9.81):
        
        # Gestione Dati
        self.input_data = None
        self.output_data = None
        
        if train_data_input is not None and train_data_output is not None:
            t_tensor = torch.as_tensor(train_data_input).float()
            y_tensor = torch.as_tensor(train_data_output).float()
            
            if t_tensor.ndim == 1:
                t_tensor = t_tensor.unsqueeze(1)
            
            self.input_data = LabelTensor(t_tensor, labels=['t'])
            # Mappiamo le label coerentemente con dynamics:
            self.output_data = LabelTensor(y_tensor, labels=['q1', 'q2', 'q1_dot', 'q2_dot'])

        # Parametri Fisici
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.b1 = b1
        self.b2 = b2

        super().__init__()

    @property
    def conditions(self):
        conds = {}

        # --- Physics Loss (ODE) ---
        
        # build domain samples using the input data that contains time points
        
        #domain = CartesianDomain({'t': [0, max(self.input_data)]})
        #domain_samples = domain.sample(n=1000, mode='random')
        
        conds['physics'] = Condition(input=self.input_data, 
                                     equation=Equation(self.ode_system))

        # --- Data Loss ---
        if self.input_data is not None:
            conds['data'] = InputTargetCondition(input=self.input_data, 
                                                 target=self.output_data)
            
        # --- Initial Conditions ---
        # enforce the penalty constrain that y^(t=0) = y0
        if self.input_data is not None:
            t0_mask = (self.input_data['t'] == 0).squeeze()
            if t0_mask.any():
                input_t0 = self.input_data[t0_mask]
                target_t0 = self.output_data[t0_mask]
                
                conds['initial'] = InputTargetCondition(input=input_t0,
                                                        target=target_t0)
        
        return conds

    @property
    def output_variables(self):
        return ['q1', 'q2', 'q1_dot', 'q2_dot']

    @property
    def temporal_domain(self):
        return CartesianDomain({'t': [0, max(self.input_data)]})

    def ode_system(self, input_, output_):
        # 1. Estrazione variabili e Gradienti automatici
        t_var = input_.extract(['t'])
        t_var.requires_grad_(True)
        
        # Variabili predette dalla rete (q e q_dot)
        q1 = output_.extract(['q1'])
        q2 = output_.extract(['q2'])
        q1_dot = output_.extract(['q1_dot'])
        q2_dot = output_.extract(['q2_dot'])

        # Derivate calcolate dalla rete (d_theta/dt e d_omega/dt)
        dt_q1 = grad(q1, t_var)
        dt_q2 = grad(q2, t_var)
        dt_q1_dot = grad(q1_dot, t_var) # Questo deve matchare q1_ddot fisico
        dt_q2_dot = grad(q2_dot, t_var) # Questo deve matchare q2_ddot fisico

        # 2. Parametri fisici
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        b1, b2 = self.b1, self.b2

        # 3. Costruzione Matrici (M, C, G, D) in PyTorch
        # Nota: q2 è l'angolo RELATIVO in questa formulazione (visto G con sin(q1+q2))
        
        # --- Matrice d'Inerzia M(q) ---
        # M11
        M00 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * torch.cos(q2)
        # M12
        M01 = m2 * l2**2 + m2 * l1 * l2 * torch.cos(q2)
        # M21 (Simmetrica)
        M10 = M01
        # M22
        M11 = m2 * l2**2 # Scalare, ma in tensore broadcasted
        
        # Determinante di M (per l'inversione)
        det_M = M00 * M11 - M01 * M10

        # --- Vettore Coriolis C(q, q_dot) ---
        # Termine comune
        term_c = m2 * l1 * l2 * torch.sin(q2)
        C0 = -term_c * (2 * q1_dot + q2_dot) * q2_dot
        C1 = term_c * q1_dot**2

        # --- Matrice Smorzamento D ---
        D0 = b1 * q1_dot
        D1 = b2 * q2_dot

        # --- Vettore Gravità G(q) ---
        G0 = (m1 + m2) * g * l1 * torch.sin(q1) + m2 * g * l2 * torch.sin(q1 + q2)
        G1 = m2 * g * l2 * torch.sin(q1 + q2)

        # 4. Risoluzione del sistema M * q_ddot = u - C - D - G
        # Consideriamo input u = 0
        
        # RHS (Right Hand Side) = - C - D - G
        b0 = -C0 - D0 - G0
        b1 = -C1 - D1 - G1

        # Risolviamo M * [acc1, acc2]^T = [b0, b1]^T usando Cramer (inversione esplicita 2x2)
        # q1_ddot_phys = (M11 * b0 - M01 * b1) / det_M
        # q2_ddot_phys = (-M10 * b0 + M00 * b1) / det_M
        
        acc_phys_1 = (M11 * b0 - M01 * b1) / det_M
        acc_phys_2 = (-M10 * b0 + M00 * b1) / det_M

        # 5. Calcolo Residui
        # Residuo velocità: la derivata dell'angolo deve essere uguale alla velocità predetta
        res_vel_1 = dt_q1 - q1_dot
        res_vel_2 = dt_q2 - q2_dot
        
        # Residuo accelerazione: la derivata della velocità deve essere uguale all'accelerazione fisica
        res_acc_1 = dt_q1_dot - acc_phys_1
        res_acc_2 = dt_q2_dot - acc_phys_2

        # 6. Ritorno unico tensore
        return torch.hstack([res_vel_1, res_vel_2, res_acc_1, res_acc_2])

