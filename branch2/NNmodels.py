

import torch.nn as nn
import torch
from torchdiffeq import odeint_adjoint as odeint
#from torch.autograd import grad
from torch.func import functional_call, vmap, grad, jacrev

from solver import solver
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim)
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(x + self.main(x))

class TrigLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: shape (B, 4) -> (B, 6)
        x_trig = torch.zeros((x.shape[0], 6), device=x.device)
        x_trig[:, 0] = torch.cos(x[:, 0])
        x_trig[:, 1] = torch.sin(x[:, 0])
        x_trig[:, 2] = torch.cos(x[:, 1])
        x_trig[:, 3] = torch.sin(x[:, 1])
        x_trig[:, 4] = x[:, 2]
        x_trig[:, 5] = x[:, 3]
        return x_trig


class BaselineFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, activation=nn.ReLU()):
        super(BaselineFNN, self).__init__()

        self.trig = TrigLayer()
        self.input = nn.Linear(6, hidden_dim)
        self.activation = activation

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation) for _ in range(1)]
        )
        
        self.output = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, t, x):
        
        x_dot = x[:, 2:]  # extract velocities
        # x: shape (B, 4)
        x = self.trig(x)
        x = self.input(x)           # B, C
        x = self.activation(x)      # B, C
        x = self.blocks(x)          # B, C
        x_dot_dot = self.output(x)          # B, C

        # at this point x is the acceleration, we would like to have the entire x_dot: x1_dot, x1_dot_dot, x2_dot, x2_dot_dot        
        return torch.cat([x_dot, x_dot_dot], dim=1)

class DeepFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden, out_dim, activation=nn.ReLU()):
        super(DeepFNN, self).__init__()

        self.input = nn.Linear(6, hidden_dim)
        self.trig = TrigLayer()

        self.activation = activation
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation) for _ in range(n_hidden)]
        )
    
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, t, x):
        x_dot = x[:, 2:]  # extract velocities
        x = self.trig(x)
        x = self.input(x)
        x = self.activation(x)
        x = self.blocks(x)      
        x_dot_dot = self.output(x)
  
        return torch.cat([x_dot, x_dot_dot], dim=1)
    


# class LagrangianNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, mean, std, activation=nn.Softplus()):
#         super(LagrangianNN, self).__init__()
        
#         # input_dim should be 2 * degrees_of_freedom (e.g., 4 for double pendulum)
#         self.dof = input_dim // 2
        
#         # TrigLayer expands q (dof) -> (2*dof) via cos/sin, plus keeps q_dot (dof)
#         # Total input to MLP = 2*dof (trig positions) + dof (velocities) = 3*dof
#         self.trig_output_dim = 2 * self.dof + self.dof 
        
#         self.mean = mean
#         self.std = std

#         self.mlp = nn.Sequential(
#             nn.Linear(self.trig_output_dim, hidden_dim),
#             activation,
#             ResidualBlock(hidden_dim, activation),
#             ResidualBlock(hidden_dim, activation),
#             nn.Linear(hidden_dim, 1, bias=False) # L is a scalar, usually no bias on last layer
#         ) 
#         self._reset_parameters()

#     def _reset_parameters(self):
#         """
#         Strict port of the JAX 'custom_init'.
        
#         JAX Logic:
#         - n = max(matrix_dimensions)
#         - base_std = 1 / sqrt(n)
#         - First layer: scale by 2.2
#         - Middle layers: scale by 0.58 (tuned for Softplus)
#         - Last layer: scale by n (resulting in std = sqrt(n))
#         """
#         # Filter only the linear layers to handle indices correctly
#         linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
#         num_layers = len(linear_layers)

#         for i, m in enumerate(linear_layers):
#             # Biases always zero
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

#             n = max(m.weight.shape) 
            
#             std = 1.0 / np.sqrt(n)

#             if i == 0:
#                 std *= 2.2
#             elif i == num_layers - 1:
#                 # The paper multiplies by n here, cancelling the sqrt(n) divisor
#                 # Result: std = sqrt(n)
#                 std *= n 
#             else:
#                 std *= 0.58

#             nn.init.normal_(m.weight, mean=0.0, std=std)

#     def normalize(self, x):
#         return (x - self.mean) / self.std
    
    
#     def forward(self, t, x, track_diagnostics=False):
#         # x shape: (Batch, 2*dof) -> [q, q_dot]
        
#         with torch.set_grad_enabled(True):
                        
#             x = x.requires_grad_(True)

#             # 1. Split state
#             q = x[:, :self.dof] 
#             q_dot = x[:, self.dof:]
#             q_dot_norm = self.normalize(q_dot)

#             # 2. Trig Encoding for q only
#             q_cos = torch.cos(q)
#             q_sin = torch.sin(q)
#             x_in = torch.cat([q_cos, q_sin, q_dot_norm], dim=1)

#             # 3. Compute Lagrangian
#             L_per_sample = self.mlp(x_in)
#             L = L_per_sample.sum() # Sum for batched gradient computation

#             # 4. Gradients
#             grads = grad(L, x, create_graph=True)[0]
#             dL_dq = grads[:, :self.dof]
#             dL_dv = grads[:, self.dof:]

#             # 5. Hessian (d2L / dv2) and Mixed Partial (d2L / dq dv)
#             hess_v_v = []
#             hess_v_q = []
            
#             for i in range(self.dof):
#                 # Gradients of the i-th velocity component of the Lagrangian w.r.t x
#                 grad_dL_dvi = grad(dL_dv[:, i].sum(), x, create_graph=True)[0]
#                 hess_v_q.append(grad_dL_dvi[:, :self.dof]) # d(dL/dvi)/dq
#                 hess_v_v.append(grad_dL_dvi[:, self.dof:]) # d(dL/dvi)/dv
            
#             # Stack to create matrices (Batch, DOF, DOF)
#             H_vv = torch.stack(hess_v_v, dim=1) 
#             H_vq = torch.stack(hess_v_q, dim=1) 

#             # --- ANOMALY DETECTION HOOK ---
#             if track_diagnostics:
#                 # we compute the Hamiltonian using Legendre transform
#                 H = self.compute_hamiltonian(dL_dv, q_dot, L_per_sample)
                
#                 self.last_diagnostics = {
#                     "mass_matrix": H_vv.detach().cpu(), # shape (B, D, D)
#                     "energy": H.detach().cpu(), # shape (B,),
#                     "force_q": dL_dq.detach().cpu(), # shape (B, D)
#                 }

#             # 6. Euler-Lagrange Equation: 
#             epsilon = 1e-4 * torch.eye(self.dof, device=x.device).unsqueeze(0)
#             H_vv = H_vv + epsilon

            
#             term_2 = torch.einsum('bij,bj->bi', H_vq, q_dot)
#             rhs = dL_dq - term_2
            
#             # 7. Solve for q_ddot using linalg solve for efficiency and stability
#             q_ddot = torch.linalg.solve(H_vv, rhs)
           
            
#             return torch.cat([q_dot, q_ddot], dim=1)
    
#     def compute_hamiltonian(self, dL_dv, q_dot, L_per_sample):
#         """
#         Returns the Learned Hamiltonian (Total Energy) for the batch. It uses the Legendre transform:
#         H = (dL/dv) * v - L
#         """
        
#         # H = p*v - L
#         # Element-wise mult and sum over DOF dimension
#         term_1 = (dL_dv * q_dot).sum(dim=1) 
            
#         H = term_1 - L_per_sample.squeeze()  # shape (B,)
#         return H

class ClassicODE(nn.Module):
    def __init__(self, func):
        super(ClassicODE, self).__init__()
        self.func = func
    
    def forward(self, z0, T, method='rk4'):
        # z0: initial state (tensor)
        # t_span: times at which to evaluate the solution (tensor)
        # method: ODE solver (e.g., 'dopri5', 'rk4', 'adams'). 'dopri5' is robust.
        
        # The odeint function integrates the system defined by self.func 
        # starting from z0 over the times specified in t_span.
        return solver(self.func, z0, T = T, method=method)


class NeuralODE(nn.Module):
    # 'dopri5' is too slow for random initialized Hamiltonian/Lagrangian networks
    def __init__(self, func, method='rk4'):
        super(NeuralODE, self).__init__()
        self.func = func
        self.method = method

    def forward(self, z0, t_span):
        return odeint(self.func, z0, t_span, 
                      method=self.method)




class NormalizationWrapper(nn.Module):
    def __init__(self, func, x_dot_mean, x_dot_std, x_dot_dot_mean, x_dot_dot_std):
        super().__init__()
        self.func = func
        
        # Register buffers so they are saved with the model state_dict
        # but are not updated by the optimizer
        self.register_buffer('x_dot_mean', x_dot_mean)
        self.register_buffer('x_dot_std', x_dot_std)
        self.register_buffer('x_dot_dot_mean', x_dot_dot_mean)
        self.register_buffer('x_dot_dot_std', x_dot_dot_std)

    def normalize(self, x):
        return (x - self.x_dot_mean) / self.x_dot_std

    def denormalize_acceleration(self, x_dot_dot_norm):
        return x_dot_dot_norm * self.x_dot_dot_std + self.x_dot_dot_mean

    def forward(self, t, x):        
        # 1. Normalize input for the Network
        pos = x[..., :2]
        vel = x[..., 2:]
        
        vel_norm = self.normalize(vel)
        x_norm = torch.cat([pos, vel_norm], dim=-1)
                
        # 2. Network Prediction (Normalized Space)
        pred_norm = self.func(t, x_norm)
        acc_norm = pred_norm[:, 2:] 
        
        # 3. De-normalize Acceleration
        acc_real = self.denormalize_acceleration(acc_norm)
        
        # 4. Return [velocity, acceleration] in REAL units
        # Note: In a Neural ODE, the output of the function must be the derivative 
        # of the input state. Input is [q, q_dot], Output must be [q_dot, q_ddot].
        # We use the REAL velocity from input x, not the normalized one.
        
        return torch.cat([vel, acc_real], dim=1)
    

class LagrangianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, mean, std, activation=nn.Softplus()):
        super(LagrangianNN, self).__init__()
        
        self.dof = input_dim // 2
        self.trig_output_dim = 2 * self.dof + self.dof 
        
        # FIX: Clone/detach to prevent UserWarnings about copying tensors
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone().detach())
        self.register_buffer("inv_std", 1.0 / torch.as_tensor(std, dtype=torch.float32).clone().detach())

        self.mlp = nn.Sequential(
            nn.Linear(self.trig_output_dim, hidden_dim),
            activation,
            ResidualBlock(hidden_dim, activation),
            ResidualBlock(hidden_dim, activation),
            nn.Linear(hidden_dim, 1, bias=False) 
        ) 
        self._reset_parameters()

    def _reset_parameters(self):
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        num_layers = len(linear_layers)

        for i, m in enumerate(linear_layers):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            n = max(m.weight.shape) 
            std = 1.0 / np.sqrt(n)

            if i == 0:
                std *= 2.2
            elif i == num_layers - 1:
                std *= n 
            else:
                std *= 0.58

            nn.init.normal_(m.weight, mean=0.0, std=std)

    def _lagrangian_single(self, params, x):
        """
        Computes L for a SINGLE sample. 
        Designed to be used with vmap/jacrev.
        """
        # 1. Split state
        q = x[:self.dof] 
        q_dot = x[self.dof:]
        
        # Optimization: Pre-computed inverse std
        q_dot_norm = (q_dot - self.mean) * self.inv_std

        # 2. Trig Encoding
        q_cos = torch.cos(q)
        q_sin = torch.sin(q)
        x_in = torch.cat([q_cos, q_sin, q_dot_norm], dim=0)

        # 3. Compute Lagrangian (Functional call uses passed params)
        # We target self.mlp explicitly
        return functional_call(self.mlp, params, (x_in,)).squeeze()

    def forward(self, t, x, track_diagnostics=False):
        # FIX: Extract params from the submodule (self.mlp) directly
        params = dict(self.mlp.named_parameters())
        
        # --- 1. Define Derivatives Functions ---
        
        def get_L(x_val):
            return self._lagrangian_single(params, x_val)
        
        get_grad_L = grad(get_L) 

        def get_dL_dv(x_val):
            g = get_grad_L(x_val)
            return g[self.dof:] # Return only velocity part

        # --- 2. Compute Quantities using vmap ---
        
        # vmap ensures we process the whole batch in parallel
        # jacrev computes the full Jacobian of dL/dv w.r.t x -> size (dof, 2*dof)
        # This replaces the loop!
        compute_batch_hessian = vmap(jacrev(get_dL_dv))
        compute_batch_grad = vmap(get_grad_L)
        
        # Execute vectorized ops
        hessian_block = compute_batch_hessian(x) # (Batch, dof, 2*dof)
        grads = compute_batch_grad(x)
        
        # --- 3. Extract Matrices ---
        
        # H_vq: Partial of dL/dv w.r.t q (first dof columns)
        H_vq = hessian_block[:, :, :self.dof] 
        # H_vv: Partial of dL/dv w.r.t v (last dof columns) - The Mass Matrix
        H_vv = hessian_block[:, :, self.dof:] 
        
        dL_dq = grads[:, :self.dof]

        # --- 4. Diagnostics (Optional) ---
        if track_diagnostics:
            dL_dv = grads[:, self.dof:]
            q_dot = x[:, self.dof:]
            L = vmap(get_L)(x)
            H = (dL_dv * q_dot).sum(dim=1) - L
            
            self.last_diagnostics = {
                "mass_matrix": H_vv.detach(),
                "energy": H.detach(),
                "force_q": dL_dq.detach(),
            }

        # --- 5. Euler-Lagrange Solution ---
        q_dot = x[:, self.dof:]
        
        # Solve: dL/dq - H_vq @ q_dot = H_vv @ q_ddot
        term_2 = torch.einsum('bij,bj->bi', H_vq, q_dot)
        rhs = dL_dq - term_2
        
        # Optimization: Use Cholesky solve for PD matrix
        eps_eye = torch.eye(self.dof, device=x.device).unsqueeze(0) * 1e-4
        
        q_ddot = torch.linalg.solve(H_vv+eps_eye, rhs)

        return torch.cat([q_dot, q_ddot], dim=1)
    


class StructuredLNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, mean, std, activation=nn.Softplus()):
        super(StructuredLNN, self).__init__()
        
        self.dof = input_dim // 2
        
        # Buffers for normalization
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone().detach())
        self.register_buffer("inv_std", 1.0 / torch.as_tensor(std, dtype=torch.float32).clone().detach())

        # --- 1. Potential Net V(q) ---
        self.potential_net = nn.Sequential(
            nn.Linear(2 * self.dof, hidden_dim),
            activation,
            ResidualBlock(hidden_dim, activation),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # --- 2. Mass Net M(q) ---
        self.mass_output_dim = (self.dof * (self.dof + 1)) // 2
        
        self.mass_net = nn.Sequential(
            nn.Linear(2 * self.dof, hidden_dim),
            activation,
            ResidualBlock(hidden_dim, activation),
            nn.Linear(hidden_dim, self.mass_output_dim)
        )


        # --- 3. Friction Net D(q, q_dot) ---
        # Outputs a diagonal damping coefficient vector. 
        # Input: 2*DOF (Trig q) + DOF (q_dot) = 3 * DOF
        friction_input_dim = (2 * self.dof) + self.dof
        
        self.friction_net = nn.Sequential(
            nn.Linear(friction_input_dim, hidden_dim),
            activation,
            ResidualBlock(hidden_dim, activation),
            nn.Linear(hidden_dim, self.dof) # Output one damping coef per DOF
        )
        
        # Bias for diagonal elements (sized to match full output to avoid broadcast errors)
        self.register_parameter("mass_diag_bias", nn.Parameter(torch.zeros(self.mass_output_dim)))
        
        # --- Pre-compute Indices for Reconstruction ---
        tril_indices = torch.tril_indices(row=self.dof, col=self.dof, offset=0)
        self.register_buffer("tril_row", tril_indices[0])
        self.register_buffer("tril_col", tril_indices[1])
        
        # FIX 1: Compute FLATTENED indices for use with scatter
        # Row * Width + Col = Flat Index
        self.register_buffer("tril_idx_flat", self.tril_row * self.dof + self.tril_col)
        
        # Mask for diagonal elements
        self.register_buffer("diag_mask", (self.tril_row == self.tril_col))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.mass_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.potential_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _lagrangian_single(self, params_mass, params_pot, diag_bias, x):
        """
        Computes L = T - V for a single sample.
        """
        q = x[:self.dof] 
        q_dot = x[self.dof:]
        
        q_dot_norm = (q_dot - self.mean) * self.inv_std
        
        q_cos = torch.cos(q)
        q_sin = torch.sin(q)
        x_in = torch.cat([q_cos, q_sin], dim=0)

        # 1. Potential V(q)
        V = functional_call(self.potential_net, params_pot, (x_in,)).squeeze()

        # 2. Mass Matrix M(q)
        l_vec = functional_call(self.mass_net, params_mass, (x_in,)).squeeze()
        
        # Apply bias and softplus to diagonals only
        effective_bias = diag_bias * self.diag_mask
        l_vec_biased = l_vec + effective_bias
        
        l_vec_mod = torch.where(
            self.diag_mask, 
            torch.nn.functional.softplus(l_vec_biased), 
            l_vec_biased
        )

        # FIX 2: Use functional scatter (Out-of-place) instead of index_put_ (In-place)
        # a. Create flat zero vector
        flat_zeros = torch.zeros(self.dof * self.dof, dtype=x.dtype, device=x.device)
        
        # b. Scatter values into the flat vector
        # This returns a NEW tensor, satisfying vmap's requirements
        L_flat = flat_zeros.scatter(0, self.tril_idx_flat, l_vec_mod)
        
        # c. Reshape into matrix
        L_mat = L_flat.view(self.dof, self.dof)
        
        # d. M = L * L^T
        M = L_mat @ L_mat.T

        # 3. Kinetic Energy
        T = 0.5 * torch.dot(q_dot, M @ q_dot)

        return T - V


    def get_friction_force(self, x):
        """
        Computes non-conservative forces.
        We do this outside the Lagrangian differentiation path.
        """
        q = x[:, :self.dof]
        q_dot = x[:, self.dof:]
        
        # 1. Normalize Inputs
        q_dot_norm = (q_dot - self.mean) * self.inv_std
        q_cos = torch.cos(q)
        q_sin = torch.sin(q)
        
        # Input to friction net: [cos(q), sin(q), normalized_q_dot]
        fric_in = torch.cat([q_cos, q_sin, q_dot_norm], dim=1)
        
        # 2. Predict Damping Coefficients
        # We use Softplus to ensure damping is always positive (dissipating energy)
        damping_coefficients = torch.nn.functional.softplus(self.friction_net(fric_in))
        
        # 3. Calculate Force: F = -D * q_dot
        # We multiply by un-normalized q_dot for correct physics units
        friction_force = -damping_coefficients * q_dot
        
        return friction_force


    def forward(self, t, x, track_diagnostics=False):
        params_mass = dict(self.mass_net.named_parameters())
        params_pot = dict(self.potential_net.named_parameters())
        
        current_bias = self.mass_diag_bias
            
        def get_L(x_val):
            return self._lagrangian_single(params_mass, params_pot, current_bias, x_val)
        
        get_grad_L = grad(get_L)

        def get_dL_dv(x_val):
            g = get_grad_L(x_val)
            return g[self.dof:] 


        
        # Vectorized Computations
        compute_batch_hessian = vmap(jacrev(get_dL_dv))
        compute_batch_grad = vmap(get_grad_L)
        
        hessian_block = compute_batch_hessian(x)
        grads = compute_batch_grad(x)
        
        H_vq = hessian_block[:, :, :self.dof] 
        H_vv = hessian_block[:, :, self.dof:] 
        dL_dq = grads[:, :self.dof]

        # --- 2. Friction Dynamics (Non-Conservative) ---
        F_friction = self.get_friction_force(x)


        dL_dv = grads[:, self.dof:]
        q_dot = x[:, self.dof:]
        L_val = vmap(get_L)(x)
        energy = (dL_dv * q_dot).sum(dim=1) - L_val

        if track_diagnostics:
            self.last_diagnostics = {
                "mass_matrix": H_vv.detach(),
                "energy": energy.detach(),
                "force_q": dL_dq.detach(),
            }

    
        q_dot = x[:, self.dof:]
        term_2 = torch.einsum('bij,bj->bi', H_vq, q_dot)
        rhs = dL_dq - term_2 + F_friction

        
        eps_eye = torch.eye(self.dof, device=x.device).unsqueeze(0) * 1e-6
        L_cholesky = torch.linalg.cholesky(H_vv + eps_eye)
        q_ddot = torch.cholesky_solve(rhs.unsqueeze(-1), L_cholesky).squeeze(-1)

        return torch.cat([q_dot, q_ddot], dim=1)
    



class ProbabilisticBaselineFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, activation=nn.SiLU()):
        super().__init__()
        self.trig = TrigLayer()
        self.input = nn.Linear(6, hidden_dim)
        self.activation = activation
        

        self.dropout = nn.Dropout(p=0.1)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, activation),
            self.dropout,
            ResidualBlock(hidden_dim, activation),
            self.dropout,
        )
        
        # Output dimension is doubled: [Mean, Log_Variance]
        self.output = nn.Linear(hidden_dim, out_dim * 2)

    def forward(self, t, x):
        # x: (B, 4)
        x_in = self.trig(x)
        h = self.input(x_in)
        h = self.activation(h)
        h = self.blocks(h)
        
        raw_out = self.output(h)
        
        # Split into Mean and Log-Variance
        # We assume the output columns are [mu_1, mu_2, ..., logvar_1, logvar_2, ...]
        mu, log_var = torch.chunk(raw_out, 2, dim=-1)
        
        # Return tuple (expected by the ProbabilisticWrapper)
        return mu, log_var


class ProbabilisticWrapper(nn.Module):
    def __init__(self, func, stats):
        super().__init__()
        self.func = func
        
        # Register buffers (saved with model)
        self.register_buffer('x_dot_mean', stats['x_dot_mean'])
        self.register_buffer('x_dot_std', stats['x_dot_std'])
        self.register_buffer('x_dot_dot_mean', stats['x_dot_dot_mean'])
        self.register_buffer('x_dot_dot_std', stats['x_dot_dot_std'])

    def normalize(self, vel):
        return (vel - self.x_dot_mean) / self.x_dot_std

    def denormalize_acc(self, acc_norm_mu, acc_norm_log_var):
        # Denormalize Mean
        acc_real_mu = acc_norm_mu * self.x_dot_dot_std + self.x_dot_dot_mean
        
        # Denormalize Log-Variance
        # log(var_real) = log(var_norm * std^2) = log_var_norm + 2*log(std)
        acc_real_log_var = acc_norm_log_var + 2 * torch.log(self.x_dot_dot_std)

        acc_real_log_var = torch.clamp(acc_real_log_var, min=-10.0, max=10.0)

        return acc_real_mu, acc_real_log_var

    def forward(self, t, x, return_distribution=True):
        """
        Args:
            return_distribution (bool): If True, returns (vel, mu, log_var). If False, returns (vel, mu).
            output_normalized (bool): If True, returns raw network outputs (normalized). 
                                      If False, returns physical units (m/s^2).
        """
        pos = x[..., :2]
        vel = x[..., 2:]
        
        # 1. Normalize Inputs
        vel_norm = self.normalize(vel)
        x_norm = torch.cat([pos, vel_norm], dim=-1)
        
        # 2. Network Prediction (Always Normalized)
        mu_norm, log_var_norm = self.func(t, x_norm)
        
        mu_acc, log_var_acc = self.denormalize_acc(mu_norm, log_var_norm)
    

        if return_distribution:
            return vel, mu_acc, log_var_acc
                
        else:
            # SOLVER MODE: Return single deterministic vector [v, a] in REAL units
            return torch.cat([vel, mu_acc], dim=1)