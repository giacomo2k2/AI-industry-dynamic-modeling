import os
import re

import numpy as np
from matplotlib import pyplot as plt
import torch



def generate_run_dir(base_dir, solver_name, func_name, frict_type):
    pattern = re.compile(rf"{solver_name}_{func_name}_{frict_type}_(\d+)$")

    # Create base dir if not exists
    os.makedirs(base_dir, exist_ok=True)

    # List all existing runs
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and pattern.match(d)
    ]

    # Get max numeric ID
    if existing:
        last_id = max(int(pattern.match(d).group(1)) for d in existing)
    else:
        last_id = 0

    new_id = last_id + 1
    run_name = f"{solver_name}_{func_name}_{frict_type}_{new_id:03d}"
    run_dir = os.path.join(base_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir, new_id

def history_plot(train_losses, test_losses, run_dir, solver_name, func_name, frict_type, figsize=(10, 6)):

    fig = plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss History - {solver_name} - {func_name} ({frict_type})')
    fig.savefig(f"{run_dir}/loss_history.png")
    plt.close(fig)


def state_dot_step_plot(x_dot, x_dot_pred, ids, run_dir, solver_name, func_name, frict_type, x_std=None, figsize=(10, 8)):
    """
    Plots state derivatives. Applies confidence intervals only to the 
    acceleration terms (indices 2 and 3).
    
    Args:
        x_std: Optional array of shape (Time, IDs, 4) containing standard deviations.
               (Data at indices 0 and 1 is ignored; indices 2 and 3 are used for plotting).
    """
    # Create directory if it doesn't exist
    os.makedirs(f"{run_dir}/q_dot_comparison", exist_ok=True)
    
    # Create a time/step axis
    steps = np.arange(x_dot.shape[0])
    
    for id in ids:
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        
        # --- Helper function ---
        def plot_component(axis, ground_truth, prediction, std_dev, dim_idx, label_base, show_confidence):
            # Plot Ground Truth
            axis.plot(steps, ground_truth[:, id, dim_idx], label=f'{label_base}')
            
            # Plot Prediction
            axis.plot(steps, prediction[:, id, dim_idx], label=f'{label_base}_gt', linestyle='--')
            
            # Plot Confidence Interval ONLY if requested and std_dev is present
            if show_confidence and std_dev is not None:
                mean = prediction[:, id, dim_idx]
                sigma = std_dev[:, id, dim_idx]
                
                axis.fill_between(steps, 
                                  mean - 2 * sigma, 
                                  mean + 2 * sigma, 
                                  color='orange', 
                                  alpha=0.2, 
                                  label='95% Conf')
            axis.legend()

        # --- Plotting ---
        
        # 1. q1_dot (Velocity) -> Index 0 -> NO Confidence
        plot_component(ax[0, 0], x_dot, x_dot_pred, x_std, 0, 'q1_dot', show_confidence=False)

        # 2. q1_dot_dot (Acceleration) -> Index 2 -> YES Confidence
        plot_component(ax[0, 1], x_dot, x_dot_pred, x_std, 2, 'q1_dot_dot', show_confidence=True)

        # 3. q2_dot (Velocity) -> Index 1 -> NO Confidence
        plot_component(ax[1, 0], x_dot, x_dot_pred, x_std, 1, 'q2_dot', show_confidence=False)

        # 4. q2_dot_dot (Acceleration) -> Index 3 -> YES Confidence
        plot_component(ax[1, 1], x_dot, x_dot_pred, x_std, 3, 'q2_dot_dot', show_confidence=True)

        plt.suptitle(f'State Derivatives Comparison - {solver_name} - {func_name} ({frict_type})')
        plt.tight_layout()
        fig.savefig(f"{run_dir}/q_dot_comparison/id_{id}.png")
        plt.close(fig)


def state_evolution_plot(x, x_pred, ids, run_dir, solver_name, func_name, frict_type, x_std=None, figsize=(10, 8)):
    os.makedirs(f"{run_dir}/state_evolution", exist_ok=True)
    steps = np.arange(x.shape[0])
    
    for id in ids:
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        
        dims = ['q1', 'q2', 'q1_dot', 'q2_dot']
        
        for i, dim_name in enumerate(dims):
            row = i // 2
            col = i % 2
            
            # Ground Truth
            ax[row, col].plot(steps, x[:, id, i], label=dim_name)
            
            # Prediction
            ax[row, col].plot(steps, x_pred[:, id, i], label=f'{dim_name}_pred', linestyle='--')
            
            # Confidence Interval
            if x_std is not None:
                mean = x_pred[:, id, i]
                sigma = x_std[:, id, i]
                ax[row, col].fill_between(steps, mean - 2*sigma, mean + 2*sigma, 
                                          color='orange', alpha=0.2, label='95% Conf')
            
            ax[row, col].legend()
            
        plt.suptitle(f'State Evolution - {solver_name}')
        plt.tight_layout()
        fig.savefig(f"{run_dir}/state_evolution/id_{id}.png")
        plt.close(fig)



def compute_energy(pendulum, x):
    T = x.shape[0]
    B = x.shape[1]
    E = np.zeros((T, B))

    # x = [q1, q2, q1_dot, q2_dot] -> [q1, q1_dot, q2, q2_dot]

    x = x[:, :, [0, 2, 1, 3]]
    
    for b in range(B):
        for t in range(T):
            _, _, E[t, b] = pendulum.energy(x[t, b, :])
    return E
   
def energy_evolution_plot(energy, energy_pred, ids, run_dir, solver_name, func_name, frict_type, figsize=(10, 6)):
    
    os.makedirs(f"{run_dir}/energy_evolution", exist_ok=True)
    
    for id in ids:
        fig = plt.figure(figsize=figsize)
        plt.plot(energy[:, id], label='True Energy')
        plt.plot(energy_pred[:, id], label='Predicted Energy')
        plt.legend()
        plt.xlabel('Time Steps')
        plt.ylabel('Energy')
        plt.title(f'Energy Evolution - {solver_name} - {func_name} ({frict_type})')
        fig.savefig(f"{run_dir}/energy_evolution/id_{id}.png")
        plt.close(fig)


def round_to_pi(x: np.ndarray) -> np.ndarray:
    """
    Wrap angles to the range [-pi, pi].
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x_norm * std + mean


def normalize_log_var(log_var: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return log_var - 2 * torch.log(std)