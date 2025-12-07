import numpy as np
import torch

def solver(x: torch.Tensor, model: torch.nn.Module,  dt: float, method: str = 'fe') -> torch.Tensor:

    if method == 'rk4':
        k1 = model(x)
        k2 = model(x + dt * k1 / 2)
        k3 = model(x + dt * k2 / 2)
        k4 = model(x + dt * k3)

        new_states = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif method == 'fe':
        new_states = x + dt * model(x)
    else:
        raise ValueError("Choose a valid method: 'fe' or 'rk'")
    return new_states


def round_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to the range [-pi, pi].
    """
    return (x + torch.pi) % (2 * torch.pi) - torch.pi
