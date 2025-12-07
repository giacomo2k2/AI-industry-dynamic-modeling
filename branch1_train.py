import torch
import pandas as pd
import numpy as np
from torch import nn
from torchdiffeq import odeint
import math
from training_utils import train_nn_model, plot_training_history, EarlyStopping, physics_loss

# 1. Architecture for the estimation of constant parameters

    # Architectures:
    # a. RNN-based architecture
    # b. UDEs

class Dyn_layer(nn.Module):
    def __init__(self,
                 b1_ref: float = 0.1,
                 fixed_b1: float = None,
                 b2_ref: float = 0.1,
                 fixed_b2: float = None, 
                 l1_ref: float = 1.0,
                 fixed_l1: float = None, 
                 l2_ref: float = 1.0,
                 fixed_l2: float = None, 
                 m1_ref: float = 1.0,
                 fixed_m1: float = None, 
                 m2_ref: float = 1.0,
                 fixed_m2: float = None,
                 ):
        
        super().__init__()
        self.b1_ref = b1_ref
        self.b2_ref = b2_ref
        self.l1_ref = l1_ref
        self.l2_ref = l2_ref
        self.m1_ref = m1_ref
        self.m2_ref = m2_ref
        self.fixed_b1 = fixed_b1
        self.fixed_b2 = fixed_b2
        self.fixed_l1 = fixed_l1
        self.fixed_l2 = fixed_l2
        self.fixed_m1 = fixed_m1
        self.fixed_m2 = fixed_m2

        if fixed_b1 is None:
            self.logb1 = nn.Parameter(torch.randn(1))
        else:
            val = math.log(fixed_b1 / self.b1_ref)
            self.register_buffer("logb1", torch.tensor([val], dtype=torch.float32))

        if fixed_b2 is None:
            self.logb2 = nn.Parameter(torch.randn(1))
        else:
            val = math.log(fixed_b2 / self.b2_ref)
            self.register_buffer("logb2", torch.tensor([val], dtype=torch.float32))
        
        if fixed_l1 is None:
            self.logl1 = nn.Parameter(torch.randn(1))
        else:
            val = math.log(fixed_l1 / self.l1_ref)
            self.register_buffer("logl1", torch.tensor([val], dtype=torch.float32))
        
        if fixed_l2 is None:
            self.logl2 = nn.Parameter(torch.randn(1))
        else:
            val = math.log(fixed_l2 / self.l2_ref)
            self.register_buffer("logl2", torch.tensor([val], dtype=torch.float32))
        
        if fixed_m1 is None:
            self.logm1 = nn.Parameter(torch.randn(1))
        else:
            val = math.log(fixed_m1 / self.m1_ref)
            self.register_buffer("logm1", torch.tensor([val], dtype=torch.float32))

        if fixed_m2 is None:
            self.logm2 = nn.Parameter(torch.randn(1))
        else:
            val = math.log(fixed_m2 / self.m2_ref)
            self.register_buffer("logm2", torch.tensor([val], dtype=torch.float32))
    
    def get_b1(self):
        return torch.exp(self.logb1) * self.b1_ref
    
    def get_b2(self):
        return torch.exp(self.logb2) * self.b2_ref
    
    def get_l1(self):
        return torch.exp(self.logl1) * self.l1_ref
    
    def get_l2(self):
        return torch.exp(self.logl2) * self.l2_ref
    
    def get_m1(self):
        return torch.exp(self.logm1) * self.m1_ref
    
    def get_m2(self):
        return torch.exp(self.logm2) * self.m2_ref
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [4,] -> [q1, q1_dot, q2, q2_dot] SINGOLO CAMPIONE
        
        assert inputs.shape == (4,), f"Expected [4], got {inputs.shape}"
        
        q1, q1_dot, q2, q2_dot = inputs  
        
        
        m1, m2 = self.get_m1(), self.get_m2()
        l1, l2 = self.get_l1(), self.get_l2()
        g = 9.81 
        b1, b2 = self.get_b1(), self.get_b2()
        
        # Termini trigonometrici SCALARI
        c2 = torch.cos(q2)
        s2 = torch.sin(q2)
        s1 = torch.sin(q1)
        s12 = torch.sin(q1 + q2)

        # === 1. Inertia Matrix M(q) [2, 2] ===
        M11 = ((m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2).squeeze()
        M12 = (m2 * l2**2 + m2 * l1 * l2 * c2).squeeze()
        M21 = M12  # Simmetrica
        M22 = (m2 * l2**2).squeeze()
        
        M = torch.stack([
            torch.stack([M11, M12]), 
            torch.stack([M21, M22])]) # Shape [2, 2]

        assert M.shape == (2, 2), f"M shape is {M.shape}, expected (2, 2)"

        # === 2. Coriolis Vector c(q, q_dot) [2, 1] ===
        c1 = (-m2 * l1 * l2 * s2 * (2 * q1_dot + q2_dot) * q2_dot).squeeze()
        c2_elem = (m2 * l1 * l2 * s2 * q1_dot**2).squeeze()
        
        c = torch.stack([c1, c2_elem]).unsqueeze(-1) # Shape [2, 1]

        assert c.shape == (2, 1), f"c shape is {c.shape}, expected (2, 1)"

        # === 3. Damping Matrix D [2, 2] ===
        # Note: If b1 and b2 are constants, this is okay. If they are learned parameters,
        # they must be registered as such (e.g., nn.Parameter).
        # Assuming b1, b2 are already tensors here.
        D = torch.stack([
            torch.stack([b1.squeeze(), torch.zeros_like(b1.squeeze())]), 
            torch.stack([torch.zeros_like(b2.squeeze()), b2.squeeze()])
        ]) # Shape [2, 2]

        assert D.shape == (2, 2), f"D shape is {D.shape}, expected (2, 2)"

        
        # === 4. Gravity Vector G [2, 1] ===
        G1 = ((m1 + m2) * g * l1 * s1 + m2 * g * l2 * s12).squeeze()
        G2 = (m2 * g * l2 * s12).squeeze()
        
        G = torch.stack([G1, G2]).unsqueeze(-1) # Shape [2, 1]

        assert G.shape == (2, 1), f"G shape is {G.shape}, expected (2, 1)"

        # === 5. q_dot vector [2, 1] ===
        q_dot_vec = torch.stack([q1_dot, q2_dot]).unsqueeze(-1) # Shape [2, 1]

        assert q_dot_vec.shape == (2, 1), f"q_dot_vec shape is {q_dot_vec.shape}, expected (2, 1)"

        # === Solve M @ q_ddot = rhs ===
        rhs = -c - (D @ q_dot_vec) - G  # [2, 1]
        
        M_batched = M.unsqueeze(0)        # Shape [1, 2, 2]
        rhs_batched = rhs.unsqueeze(0)    # Shape [1, 2, 1]
        
        q_ddot_batched = torch.linalg.solve(M_batched, rhs_batched) # Shape [1, 2, 1]
        q_ddot = q_ddot_batched.squeeze(0) # Shape [2, 1]

        # Concatena [q_dot, q_ddot] -> [4, 1] -> [4]
        # We use torch.cat([q_dot_vec, q_ddot], dim=0) to be safe, then squeeze.
        next_states = torch.cat([q_dot_vec, q_ddot], dim=0).squeeze(-1) # Shape [4]
        
        return next_states

    
class EulerNN(nn.Module):
    def __init__(self, f, auxiliary_input=False):
        super().__init__()
        self.f = f 
        self.auxiliary_input = auxiliary_input

    def integration(self, y, T, aux = None):

        # y: (time_steps, state_dim)
        # T: (time_steps, time_step)

        #print(f"  y shape: {y.shape}")
        #print(f"  T shape: {T.shape}")

        device = y.device  # Prendi device da y (già su GPU)
        dtype = y.dtype

        y = y[0]  # initial state (state_dim,)

        res = [y.clone()]

        for i in range(T.shape[0] - 1): # integrate N times
            
            t  = T[i,0]     # (1,)
            nt = T[i+1,0]   # (1,)

            if self.auxiliary_input:
                # aux: (batch, time_steps, aux_dim)
                dy = self.f((y, aux[i])) # TODO : cannot work
            else:
                dy = self.f(y)

            dy = dy.to(device=device, dtype=dtype)

            # broadcast di (nt - t) su y
            dt = (nt - t).to(device=device, dtype=dtype)
            
            # se y ha più dimensioni, estendi dt
            while dt.dim() < y.dim():
                dt = dt.unsqueeze(-1)

            # TODO runge kutta
            #print(f" y shape: {y.shape}")
            #print(f"dt shape: {dt.shape}")
            #print(f"dy shape: {dy.shape}")

            y = y + dt * dy

            res.append(y)

        # stack lungo l'asse temporale -> (batch, state_dim)
        res = torch.stack(res, dim=0)
        return res

    def forward(self, inputs):
        # inputs: (y0, T) oppure (y0, T, aux)

        # ("Inputs shape ", inputs.shape)
        # (batch_size, time_steps, |y+T+aux| )

        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]

        if self.auxiliary_input:
            y = inputs[:, :, :4]
            T =  inputs[:, :, 4:5]
            aux = inputs[:, :, 5:]
        else:
            y = inputs[:, :, :4]
            T = inputs[:, :, 4:]


        res = []

        for b in range(batch_size):
            
            res.append(self.integration(y[b], T[b], aux[b] if self.auxiliary_input else None))

        res = torch.stack(res, dim=0)

        return res
    
class Dyn_layer_batched(nn.Module):
    def __init__(self,
                 b1_ref: float = 0.5, fixed_b1: float = None,
                 b2_ref: float = 0.5, fixed_b2: float = None, 
                 l1_ref: float = 1.0, fixed_l1: float = None, 
                 l2_ref: float = 1.0, fixed_l2: float = None, 
                 m1_ref: float = 2.0, fixed_m1: float = None, 
                 m2_ref: float = 2.0, fixed_m2: float = None):
        
        super().__init__()
        # ... (Inizializzazione parametri identica a prima) ...
        self.b1_ref, self.b2_ref = b1_ref, b2_ref
        self.l1_ref, self.l2_ref = l1_ref, l2_ref
        self.m1_ref, self.m2_ref = m1_ref, m2_ref
        
        # Helper function per inizializzare i log-parametri
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
    
class EulerNN_batched(nn.Module):
    def __init__(self, f, auxiliary_input=False):
        super().__init__()
        self.f = f 
        self.auxiliary_input = auxiliary_input

    def integration(self, y0, T, aux=None):
        # y0: (Batch, state_dim)
        # T:  (Batch, time_steps, 1)
        
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
    
    
class RK4NN_batched(nn.Module):
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

        

def main():

    # Define the device, if gpu avaible take it
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Free memory cache
    if my_device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_reserved(my_device)/1e9} GB")

    # --- Train Data ---

    # Load training data
    tr_data = pd.read_csv('data/training/train_lowfrict.csv')

    # Ground truth
    m1_gt = tr_data['m1'].iloc[0]
    m2_gt = tr_data['m2'].iloc[0]
    l1_gt = tr_data['l1'].iloc[0]
    l2_gt = tr_data['l2'].iloc[0]
    b1_gt = tr_data['b1'].iloc[0]
    b2_gt = tr_data['b2'].iloc[0]

    # find the list of unique pendulum ids
    tr_ids = tr_data['id_pendulum'].unique().tolist()
    # extract only one pendulum (the first one)
    #tr_data = tr_data[tr_data['id_pendulum'] == tr_ids[0]]

    # X_train : (n_pendulums, 1000, 5)
    X_train = np.zeros((len(tr_ids), 515, 5))

    for i, pid in enumerate(tr_ids):
        pend_data = tr_data[tr_data['id_pendulum'] == pid]
        states = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()
        T = np.array(pend_data[['time_step']])
        X_train[i, :, :4] = states
        X_train[i, :, 4:] = T

    # y_train : (n_pendulums, 1000, 4)
    y_train = np.zeros((len(tr_ids), 515, 4))

    for i, pid in enumerate(tr_ids):
        pend_data = tr_data[tr_data['id_pendulum'] == pid]
        outputs = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()
        y_train[i, :, :] = outputs

    # --- Val Data ---
    
    val_data = pd.read_csv('data/training/val_lowfrict.csv')

    val_ids = val_data['id_pendulum'].unique().tolist()

    X_val = np.zeros((len(val_ids), 515, 5))

    for i, pid in enumerate(val_ids):
        pend_data = val_data[val_data['id_pendulum'] == pid]
        states = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()
        T = np.array(pend_data[['time_step']])
        X_val[i, :, :4] = states
        X_val[i, :, 4:] = T

    y_val = np.zeros((len(val_ids), 515, 4))

    for i, pid in enumerate(val_ids):
        pend_data = val_data[val_data['id_pendulum'] == pid]
        outputs = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()
        y_val[i, :, :] = outputs

    # --- Debug ---

    #print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    #print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    
    # --- Model ---

    dPend = Dyn_layer_batched(
        m1_ref=2.0, l1_ref=2, b1_ref=0.3, 
        m2_ref=2.0, l2_ref=2, b2_ref=0.3
        )

    integration = RK4NN_batched(dPend)

    # --- Loss function ---

    loss_fn = physics_loss 

    # --- Training ---

    curriculum_steps = [4, 8, 16, 32] 
    epochs_per_stage = [50, 50, 50, 50] # Aumenta le epoche quando diventa più difficile
    lrs = [1e-2, 5e-3, 1e-3, 5e-4]       # Riduci il learning rate progressivamente

    early_stopper = EarlyStopping(patience=10)

    print("Inizio Curriculum Learning...")
    
    history_full = {"loss": [], "val_loss": []}

    for stage, (steps, eps, lr) in enumerate(zip(curriculum_steps, epochs_per_stage, lrs)):
        print(f"\n--- STAGE {stage+1}: Horizon {steps} steps, LR {lr} ---")
        
        # Reset Early Stopping per ogni stage per non fermarsi troppo presto
        early_stopper = EarlyStopping(patience=15)
        
        # Chiamata al training (Nota il nuovo parametro 'horizon')
        hist = train_nn_model(
            model=integration,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            loss_fn=loss_fn,
            early_stopper=early_stopper,
            epochs=eps,
            lr=lr,
            batch_size=1024, # Batch size alto per sfruttare la GPU
            device=my_device,
            verbose=True,
            horizon=steps # <--- Parametro che passeremo al Dataset
        )
        
        # Concatena la storia per il plot finale
        if "loss" in hist:
            history_full["loss"].extend(hist["loss"])
            history_full["val_loss"].extend(hist["val_loss"])

        # Stampa parametri imparati finora
        print(f"Params attuali: m1={dPend.get_m1().item():.3f}, l1={dPend.get_l1().item():.3f}, b1={dPend.get_b1().item():.3f}")
        print(f"\nParams attuali: m2={dPend.get_m2().item():.3f}, l2={dPend.get_l2().item():.3f}, b2={dPend.get_b2().item():.3f}")
        
    print("Parameters learned: ")
    print(f"\n\tb1: {dPend.get_b1().item():.4f}, b2: {dPend.get_b2().item():.4f}")
    print(f"\tl1: {dPend.get_l1().item():.4f}, l2: {dPend.get_l2().item():.4f}")
    print(f"\tm1: {dPend.get_m1().item():.4f}, m2: {dPend.get_m2().item():.4f}")  

    print("Ground truth parameters: ")
    print(f"\n\tb1: {b1_gt:.4f}, b2: {b2_gt:.4f}")
    print(f"\tl1: {l1_gt:.4f}, l2: {l2_gt:.4f}")
    print(f"\tm1: {m1_gt:.4f}, m2: {m2_gt:.4f}\n")

    
    # --- Plot training history ---
    
    plot_training_history(history = history_full, figsize=(6,10))

if __name__ == "__main__":
    main()


# 2 Architecture for the estimation of time-varying parameters

# 3. Architecture for the digital twin model

# ideas for later: 
# 1. Customized loss in order to give more importance to everything except q1 (not that important)
# 2. tweak learning rates 
# 3. Normalize?

# N.B. time step seems ok for Runge - Kutta integration 