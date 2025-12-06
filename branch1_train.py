import torch
import pandas as pd
import numpy as np
from torch import nn
from torchdiffeq import odeint
import math
from training_utils import train_nn_model, plot_training_history, EarlyStopping

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
        # inputs: (batch_size, 4) -> [q1, q1_dot, q2, q2_dot]
        
        # 1. Unpacking (mantiene la dimensione del batch)
        # q1 avrà shape (batch_size,)
        q1 = inputs[:, 0]
        q1_dot = inputs[:, 1]
        q2 = inputs[:, 2]
        q2_dot = inputs[:, 3]
        
        m1, m2 = self.get_m1(), self.get_m2()
        l1, l2 = self.get_l1(), self.get_l2()
        g = 9.81
        b1, b2 = self.get_b1(), self.get_b2()
        
        # Pre-calcolo termini trigonometrici per chiarezza e per tutti i 32 elementi insieme
        c2 = torch.cos(q2)
        s2 = torch.sin(q2)
        s1 = torch.sin(q1)
        s12 = torch.sin(q1 + q2)

        # === 1. Inertia Matrix M(q) ===
        # Calcoliamo i singoli elementi della matrice. Ognuno sarà (batch_size,)
        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2**2 + m2 * l1 * l2 * c2
        M21 = M12 # Simmetrica
        M22 = m2 * l2**2 * torch.ones_like(q1) # Usa ones_like per avere la shape del batch

        # Usiamo torch.stack per creare la matrice (Batch, 2, 2)
        # Stack dim=1 crea le righe, dim=1 successivo unisce le righe
        M = torch.stack([
            torch.stack([M11, M12], dim=1),
            torch.stack([M21, M22], dim=1)
        ], dim=1)
        
        # === 2. Coriolis Vector c(q, q_dot) ===
        # Calcoliamo gli elementi del vettore. Ognuno sarà (batch_size,)
        c1 = -m2 * l1 * l2 * s2 * (2 * q1_dot + q2_dot) * q2_dot
        c2_elem = m2 * l1 * l2 * s2 * q1_dot**2
        
        # Creiamo il vettore colonna (Batch, 2, 1)
        c = torch.stack([c1, c2_elem], dim=1).unsqueeze(2)

        # === 3. Damping Matrix D ===
        # D deve avere shape (Batch, 2, 2) per essere compatibile con le operazioni
        zeros = torch.zeros_like(q1)
        ones = torch.ones_like(q1)
        
        D = torch.stack([
            torch.stack([b1 * ones, zeros], dim=1),
            torch.stack([zeros, b2 * ones], dim=1)
        ], dim=1)

        # === 4. Gravity Vector G ===
        G1 = (m1 + m2) * g * l1 * s1 + m2 * g * l2 * s12
        G2 = m2 * g * l2 * s12
        
        # Creiamo vettore colonna (Batch, 2, 1)
        G = torch.stack([G1, G2], dim=1).unsqueeze(2)

        # === Preparazione q_dot ===
        # Creiamo vettore colonna (Batch, 2, 1)
        q_dot_vec = torch.stack([q1_dot, q2_dot], dim=1).unsqueeze(2)

        # === Solve ===
        # Equazione: M * q_ddot + c + D * q_dot + G = tau (qui tau=0 o input esterno?)
        # Risolviamo per q_ddot: M * q_ddot = -c - D@q_dot - G
        
        # Termine noto (RHS)
        # Nota: D @ q_dot_vec esegue la moltiplicazione matriciale corretta per batch
        rhs = -c - (D @ q_dot_vec) - G
        
        # torch.linalg.solve gestisce nativamente i batch se M è (B, N, N) e rhs è (B, N, 1)
        q_ddot = torch.linalg.solve(M, rhs)
        
        # L'output sarà (Batch, 2, 1). Facciamo squeeze per tornare a (Batch, 2)
        q_ddot = q_ddot.squeeze(2)
        q_dot_vec = q_dot_vec.squeeze(2)

        #print(q_dot_vec.shape)
        #print(q_ddot.shape)
        
        next_states = torch.cat((q_dot_vec, q_ddot), dim = 1)
        
        return next_states

    
class EulerNN(nn.Module):
    def __init__(self, f, auxiliary_input=False):
        super().__init__()
        self.f = f 
        self.auxiliary_input = auxiliary_input

    def forward(self, inputs):
        # inputs: (y0, T) oppure (y0, T, aux)

        #print(inputs.shape)

        if self.auxiliary_input:
            y = inputs[:, :4]
            T =  inputs[:, 4:5]
            aux = inputs[:, 5:]
        else:
            y = inputs[:, :4]
            T = inputs[:,4]

        # y: (batch, state_dim)
        #print(y.shape)
        # T: (batch, time_steps)
        #print(T.shape)

        res = [y]

        dt  = T[0] - T[1] # TODO : pessima hardcodata, si puo fare di meglio 

        res = self.f(y) # TODO y + dt * S ANDREBBE MODIFICATA LA GROUND TRUTH
        
        return res

        # ___ NEW ___
        for i in range(T.shape[0] - 1):
            t  = T[i]     # (1,)
            nt = T[i+1]   # (1,)

            if self.auxiliary_input:
                # aux: (batch, time_steps, aux_dim)
                dy = self.f((y, aux[i])) # TODO : cannot work
            else:
                dy = self.f(y)

            # broadcast di (nt - t) su y
            dt = (nt - t)
            # se y ha più dimensioni, estendi dt
            while dt.dim() < y.dim():
                dt = dt.unsqueeze(-1)

            y = y + dt * dy
            res.append(y)

        # stack lungo l'asse temporale -> (batch, state_dim)
        res = torch.stack(res, dim=0)
        return res

def main():

    # Define the device, if gpu avaible take it
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Train Data ---

    # Load training data
    tr_data = pd.read_csv('data/training/train_lowfrict.csv')
    # find the list of unique pendulum ids
    tr_ids = tr_data['id_pendulum'].unique().tolist()
    # extract only one pendulum (the first one)
    tr_data = tr_data[tr_data['id_pendulum'] == tr_ids[0]]

    tr_states = torch.Tensor(tr_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy())
    tr_T = torch.Tensor(np.array(tr_data[['time_step']]))
    tr_outputs = torch.Tensor(tr_data[['q1_dot', 'q2_dot', 'q1_dot_dot', 'q2_dot_dot']].to_numpy())

    X_train = torch.cat((tr_states, tr_T), dim = 1)
    y_train = tr_outputs

    # --- Val Data ---

    val_data = pd.read_csv('data/training/val_lowfrict.csv')

    val_ids = val_data['id_pendulum'].unique().tolist()

    val_data = val_data[val_data['id_pendulum'] == val_ids[0]]

    val_states = torch.Tensor(val_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy())
    val_T = torch.Tensor(np.array(val_data[['time_step']]))
    val_outputs = torch.Tensor(val_data[['q1_dot', 'q2_dot', 'q1_dot_dot', 'q2_dot_dot']].to_numpy())

    X_val = torch.cat((val_states, val_T), dim = 1)
    y_val = val_outputs

    # --- Debug ---

    #print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    #print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    
    # --- Model ---

    dPend = Dyn_layer()

    euler = EulerNN(dPend)

    # --- Loss function ---

    loss_fn = nn.MSELoss()

    # --- Training ---

    early_stopper = EarlyStopping(patience=10)

    history = train_nn_model(model = euler, 
                             X_train = X_train, 
                             y_train = y_train,
                             X_val = X_val,
                             y_val = y_val, 
                             loss_fn = loss_fn, 
                             early_stopper = early_stopper,
                             epochs = 200, 
                             verbose = True,
                             device = my_device) 
    
    print("Parameters learned: ")
    print(f"\n\tb1: {dPend.get_b1().item():.4f}, b2: {dPend.get_b2().item():.4f}")
    print(f"\n\tl1: {dPend.get_l1().item():.4f}, l2: {dPend.get_l2().item():.4f}")
    print(f"\n\tm1: {dPend.get_m1().item():.4f}, m2: {dPend.get_m2().item():.4f}")  
    
    # --- Plot training history ---
    
    plot_training_history(history = history, figsize=(6,10))

if __name__ == "__main__":
    main()


# 2 Architecture for the estimation of time-varying parameters

# 3. Architecture for the digital twin model

