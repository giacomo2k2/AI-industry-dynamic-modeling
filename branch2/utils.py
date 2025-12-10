import os
import re

import numpy as np
from matplotlib import pyplot as plt



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


def state_dot_step_plot(x_dot, x_dot_pred, ids, run_dir, solver_name, func_name, frict_type, figsize=(10, 8)):
    # create directory if it doesn't exist
    os.makedirs(f"{run_dir}/q_dot_comparison", exist_ok=True)
    
    for id in ids:
    
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        ax[0, 0].plot(x_dot[:, id, 0], label='q1_dot')
        ax[0, 0].plot(x_dot_pred[:, id, 0], label='q1_dot_gt')  
        ax[0, 0].legend()  
        ax[0, 1].plot(x_dot[:, id, 2], label='q1_dot_dot')
        ax[0, 1].plot(x_dot_pred[:, id, 2], label='q1_dot_dot_gt')
        ax[0, 1].legend()
        ax[1, 0].plot(x_dot[:, id, 1], label='q2_dot')
        ax[1, 0].plot(x_dot_pred[:, id, 1], label='q2_dot_gt')
        ax[1, 0].legend()
        ax[1, 1].plot(x_dot[:, id, 3], label='q2_dot_dot')
        ax[1, 1].plot(x_dot_pred[:, id, 3], label='q2_dot_dot_gt')
        ax[1, 1].legend()   

        plt.suptitle(f'State Derivatives Comparison - {solver_name} - {func_name} ({frict_type})')
        fig.savefig(f"{run_dir}/q_dot_comparison/id_{id}.png")
        plt.close(fig)
        
def state_evolution_plot(x_true, x_pred, ids, run_dir, solver_name, func_name, frict_type, figsize=(10, 8)):
    # create directory if it doesn't exist
    os.makedirs(f"{run_dir}/state_evolution_comparison", exist_ok=True)
    
    for id in ids:
    
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        ax[0, 0].plot(x_true[:, id, 0], label='q1')
        ax[0, 0].plot(x_pred[:, id, 0], label='q1_pred')  
        ax[0, 0].legend()  
        ax[0, 1].plot(x_true[:, id, 2], label='q1_dot')
        ax[0, 1].plot(x_pred[:, id, 2], label='q1_dot_pred')
        ax[0, 1].legend()
        ax[1, 0].plot(x_true[:, id, 1], label='q2')
        ax[1, 0].plot(x_pred[:, id, 1], label='q2_pred')
        ax[1, 0].legend()
        ax[1, 1].plot(x_true[:, id, 3], label='q2_dot')
        ax[1, 1].plot(x_pred[:, id, 3], label='q2_dot_pred')
        ax[1, 1].legend()   

        plt.suptitle(f'State Evolution Comparison - {solver_name} - {func_name} ({frict_type})')
        fig.savefig(f"{run_dir}/state_evolution_comparison/id_{id}.png")
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