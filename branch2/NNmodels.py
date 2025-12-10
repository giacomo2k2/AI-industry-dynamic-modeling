import torch.nn as nn
import torch
from torchdiffeq import odeint
from torch.autograd import grad
from torch.autograd.functional import jacobian

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
    



class LagrangianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=nn.Softplus()):
        super(LagrangianNN, self).__init__()
        
        # input_dim should be 2 * degrees_of_freedom (e.g., 4 for double pendulum)
        self.dof = input_dim // 2
        
        # TrigLayer expands q (dof) -> (2*dof) via cos/sin, plus keeps q_dot (dof)
        # Total input to MLP = 2*dof (trig positions) + dof (velocities) = 3*dof
        self.trig_output_dim = 2 * self.dof + self.dof 
        
        self.mlp = nn.Sequential(
            nn.Linear(self.trig_output_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1, bias=False) # L is a scalar, usually no bias on last layer
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Strict port of the JAX 'custom_init'.
        
        JAX Logic:
        - n = max(matrix_dimensions)
        - base_std = 1 / sqrt(n)
        - First layer: scale by 2.2
        - Middle layers: scale by 0.58 (tuned for Softplus)
        - Last layer: scale by n (resulting in std = sqrt(n))
        """
        # Filter only the linear layers to handle indices correctly
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        num_layers = len(linear_layers)

        for i, m in enumerate(linear_layers):
            # Biases always zero
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            # JAX uses max(shape) for 'n'. 
            # m.weight.shape is (out_features, in_features)
            n = max(m.weight.shape) 
            
            std = 1.0 / np.sqrt(n)

            if i == 0:
                std *= 2.2
            elif i == num_layers - 1:
                # The paper multiplies by n here, cancelling the sqrt(n) divisor
                # Result: std = sqrt(n)
                std *= n 
            else:
                std *= 0.58

            # Apply Normal distribution with the calculated std
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, t, x):
        # x shape: (Batch, 2*dof) -> [q, q_dot]
        
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            
            # 1. Split state
            q = x[:, :self.dof]
            q_dot = x[:, self.dof:]

            # 2. Trig Encoding for q only
            # cos(q), sin(q)
            q_cos = torch.cos(q)
            q_sin = torch.sin(q)
            
            # Input to MLP: [cos(q), sin(q), q_dot]
            # Note: The paper sometimes feeds raw q_dot, sometimes not. 
            # We assume q is angle, q_dot is not wrapped.
            x_in = torch.cat([q_cos, q_sin, q_dot], dim=1)
            
            L = self.mlp(x_in).sum() # Sum for batched gradient computation

            # 3. Gradients
            # We need dL/dx. Since L is sum over batch, grad(L, x) gives batched grads.
            grads = grad(L, x, create_graph=True)[0]
            
            dL_dq = grads[:, :self.dof]
            dL_dv = grads[:, self.dof:]

            # 4. Hessian (d2L / dv2) and Mixed Partial (d2L / dq dv)
            # We need the Jacobian of dL_dv w.r.t x.
            # Efficient trick without loops for small systems:
            # Iterate only over the dimension of DOF (usually small, e.g., 2), not Batch
            
            hess_v_v = []
            hess_v_q = []
            
            for i in range(self.dof):
                # Gradients of the i-th velocity component of the Lagrangian w.r.t x
                grad_dL_dvi = grad(dL_dv[:, i].sum(), x, create_graph=True)[0]
                
                hess_v_q.append(grad_dL_dvi[:, :self.dof]) # d(dL/dvi)/dq
                hess_v_v.append(grad_dL_dvi[:, self.dof:]) # d(dL/dvi)/dv
            
            # Stack to create matrices (Batch, DOF, DOF)
            H_vv = torch.stack(hess_v_v, dim=1) 
            H_vq = torch.stack(hess_v_q, dim=1) 

            # 5. Euler-Lagrange Equation: 
            # H_vv * q_ddot + H_vq * q_dot = dL_dq
            # => H_vv * q_ddot = dL_dq - H_vq * q_dot
            
            epsilon = 1e-4 * torch.eye(self.dof, device=x.device).unsqueeze(0)
            H_vv = H_vv + epsilon

            # Compute RHS
            # H_vq is (B, D, D), q_dot is (B, D) -> need (B, D, 1) for matmul or use einsum
            term_2 = torch.einsum('bij,bj->bi', H_vq, q_dot)
            rhs = dL_dq - term_2

            # 6. Solve for q_ddot using linalg solve for efficiency and stability

            q_ddot = torch.linalg.solve(H_vv, rhs)

            # Return state derivative [q_dot, q_ddot]
            return torch.cat([q_dot, q_ddot], dim=1)


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
    def __init__(self, func, method='rk4', rtol=1e-3, atol=1e-3):
        super(NeuralODE, self).__init__()
        self.func = func
        self.method = method
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, z0, t_span):
        return odeint(self.func, z0, t_span, 
                      method=self.method, 
                      rtol=self.rtol, 
                      atol=self.atol)


