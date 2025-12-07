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
        # inputs: [4,] -> [q1, q1_dot, q2, q2_dot] SINGOLO CAMPIONE
        
        assert inputs.shape == (4,), f"Expected [4], got {inputs.shape}"
        
        # Unpack tensors: these still track gradients
        q1, q1_dot, q2, q2_dot = inputs  
        
        # Parametri scalari (Assuming these are also tensors, e.g., self.m1)
        # If they are not tensors, convert them once during initialization
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
        
        # 🌟 FIX 1: Use torch.stack to build the matrix while preserving the graph
        M = torch.stack([
            torch.stack([M11, M12]), 
            torch.stack([M21, M22])]) # Shape [2, 2]

        assert M.shape == (2, 2), f"M shape is {M.shape}, expected (2, 2)"

        # === 2. Coriolis Vector c(q, q_dot) [2, 1] ===
        c1 = (-m2 * l1 * l2 * s2 * (2 * q1_dot + q2_dot) * q2_dot).squeeze()
        c2_elem = (m2 * l1 * l2 * s2 * q1_dot**2).squeeze()
        
        # 🌟 FIX 2: Use torch.stack and unsqueeze to build the column vector
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
        
        # 🌟 FIX 3: Use torch.stack and unsqueeze to build the column vector
        G = torch.stack([G1, G2]).unsqueeze(-1) # Shape [2, 1]

        assert G.shape == (2, 1), f"G shape is {G.shape}, expected (2, 1)"

        # === 5. q_dot vector [2, 1] ===
        q_dot_vec = torch.stack([q1_dot, q2_dot]).unsqueeze(-1) # Shape [2, 1]

        assert q_dot_vec.shape == (2, 1), f"q_dot_vec shape is {q_dot_vec.shape}, expected (2, 1)"

        # === Solve M @ q_ddot = rhs ===
        rhs = -c - (D @ q_dot_vec) - G  # [2, 1]
        
        # 🌟 FIX 4: Add batch dimension for linalg.solve, then remove it
        # This prevents the linalg.solve dimension error from the previous traceback.
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
    tr_data = tr_data[tr_data['id_pendulum'] == tr_ids[0]]

    tr_states = tr_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()  #[y0, y1_hat, y2_hat]
    tr_T = np.array(tr_data[['time_step']])
    tr_outputs = tr_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()  #[y0, y1, y2]

    #print (tr_states.shape)
    #print(tr_T.shape)
    X_train = np.hstack((tr_states, tr_T))
    #print(X_train.shape)
    y_train = tr_outputs

    # --- Val Data ---
    
    val_data = pd.read_csv('data/training/val_lowfrict.csv')

    val_ids = val_data['id_pendulum'].unique().tolist()

    val_data = val_data[val_data['id_pendulum'] == val_ids[0]]

    val_states = val_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()
    val_T = np.array(val_data[['time_step']])
    val_outputs = val_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy()

    X_val = np.hstack((val_states, val_T))
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
                             epochs = 10, 
                             verbose = True,
                             device = my_device,
                             batch_size = 10,) 
    
    print("Parameters learned: ")
    print(f"\n\tb1: {dPend.get_b1().item():.4f}, b2: {dPend.get_b2().item():.4f}")
    print(f"\tl1: {dPend.get_l1().item():.4f}, l2: {dPend.get_l2().item():.4f}")
    print(f"\tm1: {dPend.get_m1().item():.4f}, m2: {dPend.get_m2().item():.4f}")  

    print("Ground truth parameters: ")
    print(f"\n\tb1: {b1_gt:.4f}, b2: {b2_gt:.4f}")
    print(f"\tl1: {l1_gt:.4f}, l2: {l2_gt:.4f}")
    print(f"\tm1: {m1_gt:.4f}, m2: {m2_gt:.4f}\n")

    
    # --- Plot training history ---
    
    plot_training_history(history = history, figsize=(6,10))

if __name__ == "__main__":
    main()


# 2 Architecture for the estimation of time-varying parameters

# 3. Architecture for the digital twin model

