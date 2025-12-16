import numpy as np
import torch

def solver(func: torch.nn.Module, x0: torch.Tensor, T: torch.Tensor, method: str = 'fe') -> torch.Tensor:
    steps = T.shape[0]
    x = x0.clone()
    trajectory = torch.zeros((steps, x0.shape[0], x0.shape[1]), device=x0.device)
    trajectory[0] = x0
    for i in range(1, steps):
        dt = T[i] - T[i - 1]
        x = solver_step(func, x, dt, method)
        
        trajectory[i] = x
    return trajectory


def solver_step(func: torch.nn.Module,  x: torch.Tensor, dt: float, method: str = 'fe') -> torch.Tensor:

    if method == 'rk4':
        k1 = func(t = None, x=x).clone()
        k2 = func(t = None, x=x + dt * k1 / 2).clone()
        k3 = func(t = None, x=x + dt * k2 / 2).clone()
        k4 = func(t = None, x=x + dt * k3).clone()

        new_states = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif method == 'fe':
        new_states = x + dt * func(t = None, x=x)
    else:
        raise ValueError("Choose a valid method: 'fe' or 'rk'")
    return new_states

