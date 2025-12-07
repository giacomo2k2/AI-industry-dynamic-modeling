import torch
from torch import nn
import math

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
                return nn.Parameter(torch.zeros(1) + torch.randn(1) * 0.0001)
            else:
                val = math.log(fixed_val / ref_val)
                return torch.tensor([val], dtype=torch.float32)

        self.logb1 = init_param(fixed_b1, b1_ref) if fixed_b1 is None else self.register_buffer("logb1", init_param(fixed_b1, b1_ref))
        self.logb2 = init_param(fixed_b2, b2_ref) if fixed_b2 is None else self.register_buffer("logb2", init_param(fixed_b2, b2_ref))
        self.logl1 = init_param(fixed_l1, l1_ref) if fixed_l1 is None else self.register_buffer("logl1", init_param(fixed_l1, l1_ref))
        self.logl2 = init_param(fixed_l2, l2_ref) if fixed_l2 is None else self.register_buffer("logl2", init_param(fixed_l2, l2_ref))
        self.logm1 = init_param(fixed_m1, m1_ref) if fixed_m1 is None else self.register_buffer("logm1", init_param(fixed_m1, m1_ref))
        self.logm2 = init_param(fixed_m2, m2_ref) if fixed_m2 is None else self.register_buffer("logm2", init_param(fixed_m2, m2_ref))

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