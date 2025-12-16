# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from NNmodels import ProbabilisticBaselineFNN, ProbabilisticWrapper

import os


from utils import generate_run_dir, history_plot, state_dot_step_plot, state_evolution_plot, compute_energy, energy_evolution_plot, normalize, denormalize, normalize_log_var



# Global constants
MAX_EPOCHS = 200
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
MIN_LR = 1e-5
TEST_ITER = 10
TRAIN_TEST_SPLIT = 0.8

ACTIVATION = 'SiLU'  # Options: 'ReLU', 'Tanh', 'SiLU', 'GELU', SELU, 'Softplus'
FRICT_TYPE = 'nofrict'  # Options: 'none', 'low', 'high'
OPTIMIZER_NAME = 'AdamW' # Options: 'AdamW', 'Adam'

SOLVER_NAME = 'ClassicODE'
FUNC_NAME = 'ProbabilisticBaselineFNN'

torch.set_float32_matmul_precision('high')



def train_step_probabilistic(func, dataloader, device, optimizer, criterion, stats):
    running_loss = 0.0
    
    # Use PyTorch's built-in Gaussian NLL Loss
    # full=True includes the constant term (log(2pi)), helpful for comparing models
    
    x_dot_dot_mean = stats['x_dot_dot_mean']
    x_dot_dot_std = stats['x_dot_dot_std']


    for x_batch, x_dot_batch in dataloader:
        x_batch, x_dot_batch = x_batch.to(device), x_dot_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with distribution=True
        _, acc_pred, acc_log_var_pred = func(t=None, x=x_batch, return_distribution=True)
        
        # Get true acceleration
        acc_target = x_dot_batch[:, 2:]
        
        acc_norm_target = normalize(acc_target, x_dot_dot_mean, x_dot_dot_std)

        # Normalize mu and log_var
        acc_norm_pred = normalize(acc_pred, x_dot_dot_mean, x_dot_dot_std)
        acc_log_var_norm_pred = normalize_log_var(acc_log_var_pred, x_dot_dot_std)
     
        # Convert log_var to var for the loss function
        acc_pred_var_norm = torch.exp(acc_log_var_norm_pred)
        
        # Calculate Loss
        loss = criterion(acc_norm_pred, acc_norm_target, acc_pred_var_norm)
        
        loss.backward()
        optimizer.step()      
        running_loss += loss.item()

    return running_loss / len(dataloader)


def test_step_probabilistic(func, dataloader, device, criterion, stats):
    running_loss = 0.0
    predictions = []


    x_dot_mean = stats['x_dot_mean']
    x_dot_std = stats['x_dot_std']
    x_dot_dot_mean = stats['x_dot_dot_mean']
    x_dot_dot_std = stats['x_dot_dot_std']

    func.eval()
    with torch.no_grad():
        for x_batch, x_dot_batch in dataloader:
            x_batch, x_dot_batch = x_batch.to(device), x_dot_batch.to(device)
            
            vel_pred, acc_pred, acc_log_var_pred = func(t=None, x=x_batch, return_distribution=True)
        
            acc_var_pred = torch.exp(acc_log_var_pred)

            # Get true acceleration
            acc_target = x_dot_batch[:, 2:]
            
            acc_norm_target = normalize(acc_target, x_dot_dot_mean, x_dot_dot_std)

            # Normalize mu and log_var
            acc_norm_pred = normalize(acc_pred, x_dot_dot_mean, x_dot_dot_std)
            acc_log_var_norm_pred = normalize_log_var(acc_log_var_pred, x_dot_dot_std)
     

            # Convert log_var to var for the loss function
            acc_var_norm_pred = torch.exp(acc_log_var_norm_pred)
            
            # Calculate Loss
            loss = criterion(acc_norm_pred, acc_norm_target, acc_var_norm_pred)
            
            running_loss += loss.item()
            

            predictions.append(torch.cat((vel_pred, acc_pred, acc_var_pred), dim=1).detach().cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, predictions


def monte_carlo_evolution(func, x_init, t_eval, device, num_samples=100):
    """
    Simulates trajectories by sampling from the model's acceleration distribution.
    
    Args:
        func: The trained ProbabilisticWrapper model.
        x_init: Initial states for the batch. Shape: (Batch, 4)
        t_eval: Time tensor. Shape: (T,)
        stats: Dictionary containing normalization stats.
        num_samples: Number of Monte Carlo runs per pendulum.
        
    Returns:
        x_mean: Shape (T, Batch, 4)
        x_std:  Shape (T, Batch, 4)
    """
    
    # 1. Setup Data for Vectorization
    dt = t_eval[1] - t_eval[0]
    B, D = x_init.shape
    
    # Repeat each batch item 'num_samples' times
    # New shape: (Batch * num_samples, 4)
    # E.g., [Pendulum1_run1, Pendulum1_run2... Pendulum2_run1...]
    
    current_x = x_init.repeat_interleave(num_samples, dim=0).to(device)
    
    # Storage for the full trajectory
    trajectory = [current_x.clone()]
   
    func.eval()

    def get_derivs(state):
        """
        Returns the derivative vector [vel, acc] for the given state.
        state shape: (N, 4) -> [q, q_dot]
        returns: (N, 4) -> [q_dot, q_dot_dot_sampled]
        """
        
        
        # Network Prediction (Normalized)
        vel_pred, acc_pred, acc_log_var_pred= func(t=None, x=current_x, return_distribution=True)
        
        acc_std_pred = torch.exp(0.5 * acc_log_var_pred)

        # Sample Noise
        epsilon = torch.randn_like(acc_pred)
        acc_sample = acc_pred + epsilon * acc_std_pred
        
        # Return full derivative [velocity, acceleration]
        return vel_pred, acc_sample

    # *** CRITICAL CHANGE: Force model into TRAIN mode to activate Dropout ***
    func.train()
    with torch.no_grad():
        # Loop over time steps
        for i in range(len(t_eval) - 1):
            
            # 2. Network Prediction
            # Returns normalized mean and log_var
            # Note: We ignore the network's velocity output and use the current state's velocity
            
            
            vel_pred, acc_sample = get_derivs(current_x)

            vel_new = vel_pred + acc_sample *dt

            pos_current = current_x[:, :2]
            pos_new = pos_current + vel_new * dt

            current_x = torch.cat([pos_new, vel_new], dim=1)

            

            # k1 = get_derivs(current_x)
            # k2 = get_derivs(current_x + 0.5 * dt * k1)
            # k3 = get_derivs(current_x + 0.5 * dt * k2)
            # k4 = get_derivs(current_x + dt * k3)
            
            
            # current_x = current_x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            

            trajectory.append(current_x.clone())
    func.eval()
    # 6. Reshape and Aggregate
    # Stack time: (T, Batch*Samples, 4)
    traj_tensor = torch.stack(trajectory)
    
    # Reshape to separate Batch and Samples: (T, Batch, Samples, 4)
    traj_tensor = traj_tensor.view(len(t_eval), B, num_samples, D)
    
    # Calculate Mean and Std over the 'Samples' dimension (dim=2)
    x_mean = traj_tensor.mean(dim=2)
    x_std = traj_tensor.std(dim=2)
    
    return x_mean, x_std



def main():
    # torch seed
    torch.manual_seed(42)

    # Set device    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import full csv dataset (shape: (B*T, C))
    df = pd.read_csv(f"data/processed/double_pendulum_{FRICT_TYPE}.csv")  
    
    # Metric on energy evolution
    m1 = df['m1'].iloc[0]
    m2 = df['m2'].iloc[0]
    l1 = df['l1'].iloc[0]
    l2 = df['l2'].iloc[0]
    b1 = df['b1'].iloc[0]
    b2 = df['b2'].iloc[0]

    # x and x_dot columns definition
    x_cols = ['q1', 'q2','q1_dot', 'q2_dot']
    x_dot_cols = ['q1_dot', 'q2_dot', 'q1_dot_dot', 'q2_dot_dot']
    # Extract x and x_dot (shape: (B*T, 4))
    x = []
    x_dot = []
    energy = []

    for step in df['step'].unique():
        df_step = df[df['step'] == step]
        x_step = df_step[x_cols].values.astype(np.float32)
        x_dot_step = df_step[x_dot_cols].values.astype(np.float32)
        energy_step = df_step['E'].values.astype(np.float32)
        x.append(x_step)
        x_dot.append(x_dot_step)
        energy.append(energy_step)

    # stack along time dimension 
    x = np.stack(x)             # Shape: (T, B, 4)
    x_dot = np.stack(x_dot)         # Shape: (T, B, 4)
    energy = np.stack(energy)   # Shape: (T, B)


    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)             # Shape: (T, B, 4)
    x_dot_tensor = torch.tensor(x_dot, dtype=torch.float32)     # Shape: (T, B, 4)

    # Time observations tensor
    T_obs = torch.tensor(df['time_step'].unique(), dtype=torch.float32)  # Shape: (T,)

    # Train-test split
    num_pendulums = x_tensor.shape[1]
    split_index = int(num_pendulums * TRAIN_TEST_SPLIT)

    x_train = x_tensor[:, :split_index, :]          # shape: (T, B_train, 4)
    x_dot_train = x_dot_tensor[:, :split_index, :]  # shape: (T, B_train, 4)
 
    x_test = x_tensor[:, split_index:, :]           # shape: (T, B_test, 4)
    x_dot_test = x_dot_tensor[:, split_index:, :]   # shape: (T , B_test, 4)

    x_dot_train_mean = x_train[:, :, 2:].reshape(-1, 2).mean(dim=0).to(device)
    x_dot_train_std = x_train[:, :, 2:].reshape(-1, 2).std(dim=0).to(device)
    x_dot_dot_train_mean = x_dot_train[:, :, 2:].reshape(-1, 2).mean(dim=0).to(device)
    x_dot_dot_train_std = x_dot_train[:, :, 2:].reshape(-1, 2).std(dim=0).to(device)

    stats = {
        'x_dot_mean': x_dot_train_mean,
        'x_dot_std': x_dot_train_std,
        'x_dot_dot_mean': x_dot_dot_train_mean, 
        'x_dot_dot_std': x_dot_dot_train_std
    }

    input_dim = len(x_cols)  # 4
    output_dim = len(x_dot_cols)//2

    # Create DataLoaders
    dataset_train = TensorDataset(
        x_train.reshape(-1, 4), 
        x_dot_train.reshape(-1, 4)
    )  # shape: (T*B_train, 4)  

    dataset_test = TensorDataset(
        x_test.reshape(-1, 4), 
        x_dot_test.reshape(-1, 4)
    )    # shape: (T*B_test, 4)


    dataloader = DataLoader(TensorDataset(
        x_tensor.reshape(-1, 4), 
        x_dot_tensor.reshape(-1, 4)
    ), batch_size=BATCH_SIZE, shuffle=False)


    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    
    
     # Select activation function
    match ACTIVATION:
        case 'ReLU':
            activation = nn.ReLU()
        case 'Tanh':
            activation = nn.Tanh()
        case 'SiLU':
            activation = nn.SiLU()
        case 'GELU':
            activation = nn.GELU()
        case 'SELU':
            activation = nn.SELU()
        case 'Softplus':
            activation = nn.Softplus()
        case _:
            raise ValueError(f"Unknown ACTIVATION: {ACTIVATION}")
    

    func = ProbabilisticBaselineFNN(input_dim=input_dim, out_dim=output_dim, hidden_dim=256, activation=activation)
    wrapped_func = ProbabilisticWrapper(func, stats).to(device)

    criterion = nn.GaussianNLLLoss(full=True, reduction='mean')
   
    optimizer_parameters = wrapped_func.parameters()

    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        # 'reduce-overhead' is excellent for small batches/physics loops using CUDA
        # If it crashes, fallback to mode="default"
        wrapped_func = torch.compile(wrapped_func, mode="default")
        
    # ==========================================
    
    
     # Select optimizer 
    if OPTIMIZER_NAME == 'AdamW':
        optimizer = optim.AdamW(optimizer_parameters, lr=LEARNING_RATE)
    elif OPTIMIZER_NAME == 'Adam':
        optimizer = optim.Adam(optimizer_parameters, lr=LEARNING_RATE)
    else:
        raise ValueError(f"Unknown OPTIMIZER_NAME: {OPTIMIZER_NAME}")

    train_losses = []
    test_losses = []

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=MIN_LR)

    print("---")
    for epoch in range(MAX_EPOCHS+1):
     
        train_loss = train_step_probabilistic(wrapped_func, dataloader_train, device, optimizer, criterion, stats)  
        curricular_step_size = None  # not used in ClassicODE
        scheduler.step()

        if epoch % TEST_ITER == 0:
           
            test_loss, _ = test_step_probabilistic(wrapped_func, dataloader_test, device, criterion, stats)
    
        
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch: {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.6f} | Chunk Size: {curricular_step_size}")
        
    print("---")

    # Create run directory
    run_dir, run_id = generate_run_dir('branch2/runs/', SOLVER_NAME, FUNC_NAME, FRICT_TYPE)
    os.makedirs(run_dir, exist_ok=True)


    # History plots
    history_plot(train_losses, test_losses, run_dir, solver_name=SOLVER_NAME, func_name=FUNC_NAME, frict_type=FRICT_TYPE)
    
    # get ids 10 pendulums to plot from test set
    ids = np.arange(split_index, split_index + 10)

    _, predictions = test_step_probabilistic(wrapped_func, dataloader, device, criterion, stats)
    
    
    x_dot_pred = predictions[:, :4]
    x_dot_pred = x_dot_pred.reshape(x_dot.shape)

    x_dot_dot_var_pred_flat = predictions[:, 4:]

    T, B, _ = x_dot.shape
    x_dot_dot_var_pred = x_dot_dot_var_pred_flat.reshape((T, B, 2))
    x_dot_dot_std_pred = np.sqrt(x_dot_dot_var_pred)

    x_std_for_plot = np.zeros_like(x_dot_pred)
    x_std_for_plot[:, :, 2:] = x_dot_dot_std_pred


    state_dot_step_plot(x_dot, x_dot_pred, ids=ids, x_std=x_std_for_plot,   run_dir=run_dir, solver_name=SOLVER_NAME,func_name=FUNC_NAME, frict_type=FRICT_TYPE)


    # monte carlo evolution
    print("Running Monte Carlo Evolution...")

    x0_test = x_test[0, :, :].to(device)

    x_evol_mean, x_evol_std = monte_carlo_evolution(
        wrapped_func,
        x_init=x0_test,
        t_eval=T_obs.to(device),
        device=device,
        num_samples=1000
    )

    x_evol_mean = x_evol_mean.cpu().numpy()
    x_evol_std = x_evol_std.cpu().numpy()

    x_test_np = x_test.cpu().numpy()
    mse_evolution = np.mean((x_test_np - x_evol_mean)**2)
    print(f"State Evolution MSE: {mse_evolution:.5f}")


    ids = np.arange(0, x_evol_mean.shape[1])  # all test pendulums


    state_evolution_plot(
        x_test_np, 
        x_evol_mean, 
        ids=ids, 
        run_dir=run_dir, 
        solver_name=SOLVER_NAME, 
        func_name=FUNC_NAME, 
        frict_type=FRICT_TYPE,
        x_std=x_evol_std  # Pass the calculated uncertainty
    )

    # energy_evolution_plot(energy, energy_pred, ids=ids, run_dir=run_dir, solver_name=SOLVER_NAME, func_name=FUNC_NAME, frict_type=FRICT_TYPE)

    # # Save model weights
    # torch.save(solver.state_dict(), f"{run_dir}/weights.pth")

    # recipe = {
    #     "solver_name": SOLVER_NAME,
    #     "func_name": FUNC_NAME,
    #     "friction_type": FRICT_TYPE,
    #     "batch_size": BATCH_SIZE,
    #     "learning_rate": LEARNING_RATE,
    #     "max_epochs": MAX_EPOCHS,
    #     "optimizer": OPTIMIZER_NAME,
    #     "loss_function": LOSS,
    #     "params": params,
    #     "stats": {
    #         "x_dot_mean": x_dot_train_mean.cpu().numpy().tolist(),
    #         "x_dot_std": x_dot_train_std.cpu().numpy().tolist(),
    #         "x_dot_dot_mean": x_dot_dot_train_mean.cpu().numpy().tolist(),
    #         "x_dot_dot_std": x_dot_dot_train_std.cpu().numpy().tolist()
    #     },
    #     "final_train_loss": train_losses[-1],
    #     "final_test_loss": test_losses[-1],
    #     "state_evolution_loss": float(state_evolution_loss),
    #     "evolution_metric_energy": float(energy_evolution_loss),
    #     "timestamp": pd.Timestamp.now().isoformat()
    # }

    # with open(f"{run_dir}/recipe.json", "w") as f:
    #     json.dump(recipe, f, indent=4)

    # history = {
    #     "epochs": list(range(0, MAX_EPOCHS+1, TEST_ITER)),
    #     "train_losses": train_losses,
    #     "test_losses": test_losses
    # }

    # with open(f"{run_dir}/history.json", "w") as f:
    #     json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()