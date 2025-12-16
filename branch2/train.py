# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from NNmodels import BaselineFNN, DeepFNN, LagrangianNN, ClassicODE, NeuralODE, NormalizationWrapper, StructuredLNN, ProbabilisticBaselineFNN, ProbabilisticWrapper
from models import DoublePendulum


import os
import json

from utils import generate_run_dir, history_plot, state_dot_step_plot, state_evolution_plot, compute_energy, energy_evolution_plot, normalize, denormalize
from metrics import energy_evolution_metric

# Global constants
MAX_EPOCHS  = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MIN_LR = 1e-5
TEST_ITER = 10
TRAIN_TEST_SPLIT = 0.8
MAX_CHUNK_SIZE = 15

ACTIVATION = 'SiLU'  # Options: 'ReLU', 'Tanh', 'SiLU', 'GELU', SELU, 'Softplus'

SOLVER_NAME = 'ClassicODE'  # Options: 'ClassicODE', 'NeuralODE'
FUNC_NAME = 'StructuredLNN'  # Options: 'BaselineFNN', 'DeepFNN', 'HamiltonianNN', 'LagrangianNN', 'StructuredLNN', 'ProbabilisticBaselineFNN'

FRICT_TYPE = 'nofrict'  # Options: 'none', 'low', 'high'
OPTIMIZER_NAME = 'AdamW' # Options: 'AdamW', 'Adam'
LOSS = 'MSELoss'  # Options: 'MSELoss', 'MAELoss', 'GaussianNLLLoss', 'HuberLoss'


torch.set_float32_matmul_precision('high')

def train_step(func, dataloader, device, criterion, optimizer, stats):
    # Train a single step ahead using the batchsize in the dataloader
    # Dataset: shuffled    
    running_loss = 0.0

    x_dot_dot_mean = stats['x_dot_dot_mean']
    x_dot_dot_std = stats['x_dot_dot_std']

    for x_batch, x_dot_batch in dataloader:
        x_batch, x_dot_batch = x_batch.to(device), x_dot_batch.to(device)
        
        optimizer.zero_grad()                       # Reset gradients
        x_dot_pred = func(t=None, x=x_batch)          # Forward pass

        loss = criterion(normalize(x_dot_pred[:, 2:], x_dot_dot_mean, x_dot_dot_std), normalize(x_dot_batch[:, 2:], x_dot_dot_mean, x_dot_dot_std))   # Loss
        loss.backward()                             # Backpropagation
        optimizer.step()                            # Update weights      
        running_loss += loss.item()

    return running_loss / len(dataloader)


def train_evolution(solver, device, x, x0, T, criterion, optimizer, stats):
    # Train the model to learn all the future evolution of the model
    # Model: call the solver + func
    # Use-case: Neural ODE 
    x0 = x0.to(device)
    x = x.to(device)
    T = T.to(device)

    x_dot_mean = stats['x_dot_mean']
    x_dot_std = stats['x_dot_std']


    optimizer.zero_grad()                       # Reset gradients
    
    x_pred = solver(x0, T)                       # Forward pass

    pred_pos = x_pred[..., :2]
    pred_vel = x_pred[..., 2:]
    
    target_pos = x[..., :2]
    target_vel = x[..., 2:]

    pred_vel_norm = normalize(pred_vel, x_dot_mean, x_dot_std)
    target_vel_norm = normalize(target_vel, x_dot_mean, x_dot_std)

    x_pred_norm = torch.cat([pred_pos, pred_vel_norm], dim=-1)
    x_norm_target = torch.cat([target_pos, target_vel_norm], dim=-1)


    train_loss = criterion(x_pred_norm, x_norm_target)

    
    train_loss.backward()                       # Backpropagation
    optimizer.step()                            # Update weights        

    return train_loss.item()
        
                        
def test_step(func, dataloader, device, criterion, stats):
    # Test the single step ahead training
    # Out: derivative of the next step. THen a solver integration is necessary
    func.eval()
    total_loss = 0.0

    x_dot_dot_mean = stats['x_dot_dot_mean']
    x_dot_dot_std = stats['x_dot_dot_std']

    predictions = []
    with torch.no_grad():
        for x_batch, x_dot_batch in dataloader:
            x_batch, x_dot_batch = x_batch.to(device), x_dot_batch.to(device)

            x_dot_pred = func(t=None, x=x_batch)
            
            predictions.append(x_dot_pred.detach().cpu().numpy())

            loss = criterion(normalize(x_dot_pred[:, 2:], x_dot_dot_mean, x_dot_dot_std), normalize(x_dot_batch[:, 2:], x_dot_dot_mean, x_dot_dot_std))
            total_loss += loss.item()

    predictions = np.concatenate(predictions, axis=0)
    return total_loss / len(dataloader), predictions


def test_evolution(solver, device, x, x0, T, criterion, stats):
    # Test the all time evolution
    # Out: all time step states already integred

    x_dot_mean = stats['x_dot_mean']
    x_dot_std = stats['x_dot_std']

    solver.eval()
    with torch.no_grad():
        x0 = x0.to(device)
        x = x.to(device)
        T = T.to(device)

        x_pred = solver(x0, T)

        pred_pos = x_pred[..., :2]
        pred_vel = x_pred[..., 2:]
        
        target_pos = x[..., :2]
        target_vel = x[..., 2:]

        pred_vel_norm = normalize(pred_vel, x_dot_mean, x_dot_std)
        target_vel_norm = normalize(target_vel, x_dot_mean, x_dot_std)

        x_pred_norm = torch.cat([pred_pos, pred_vel_norm], dim=-1)
        x_norm_target = torch.cat([target_pos, target_vel_norm], dim=-1)

        test_loss = criterion(x_pred_norm, x_norm_target)
    
    return test_loss.item(), x_pred

def train_step_probabilistic(func, dataloader, device, optimizer, criterion):
    running_loss = 0.0
    
    # Use PyTorch's built-in Gaussian NLL Loss
    # full=True includes the constant term (log(2pi)), helpful for comparing models
    

    for x_batch, x_dot_batch in dataloader:
        x_batch, x_dot_batch = x_batch.to(device), x_dot_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with distribution=True
        # We ignore 'vel_pred' because velocity is just a copy of input (deterministic)
        _, acc_mu, acc_log_var = func(t=None, x=x_batch, return_distribution=True)
        
        # Get true acceleration
        acc_target = x_dot_batch[:, 2:]
        
        # Convert log_var to var for the loss function
        acc_var = torch.exp(acc_log_var)
        
        # Calculate Loss
        loss = criterion(acc_mu, acc_target, acc_var)
        
        loss.backward()
        optimizer.step()      
        running_loss += loss.item()

    return running_loss / len(dataloader)

def curricular_step(epoch, max_epochs, initial_chunk_size, final_chunk_size):
    # Linear progression is safer for ODE training stability
    progress = epoch / max_epochs
    chunk_size = initial_chunk_size + (final_chunk_size - initial_chunk_size) * progress
    return int(chunk_size)


def get_batch(true_y, t, max_batch_size=32, chunk_size=20):
    """
    true_y: Your entire dataset of shape [Total_Time_Steps, State_Dim] (e.g., [1000, 2])
    t: Your time tensor [0.0, 0.01, ..., 10.0]
    """
    total_time, total_pendulums, state_dim = true_y.shape

    # decrease batch size when chunk size increases
    batch_size = min(max_batch_size, 512 // chunk_size)

    
    # 1. Select random start points (indices)
    # Ensure we don't pick an index too close to the end
    s = torch.from_numpy(
        np.random.choice(np.arange(total_time - chunk_size, dtype=np.int64), 
                         batch_size, 
                         replace=False)
    )

    b_idx = torch.from_numpy(np.random.choice(
        np.arange(total_pendulums, dtype=np.int64), batch_size, replace=True))
    
    # 2. creating the batch of initial states
    # Shape: [Batch_Size, State_Dim]
    batch_y0 = true_y[s, b_idx, :]  
    
    # 3. Create the batch of "Ground Truth" trajectories to compare against
    # We want to grab specific slices: s[0] to s[0]+chunk, s[1] to s[1]+chunk...
    batch_y_target = torch.stack([true_y[si:si+chunk_size, bi, :] for si, bi in zip(s, b_idx)], dim=1)  # Shape: [Batch_Size, Chunk_Size, State_Dim]
    
    # 4. Create the 'relative' time points for the solver
    # The solver typically needs t starting at 0 for the integration step
    dt = t[1] - t[0]
    batch_t = torch.linspace(0, (chunk_size - 1) * dt, chunk_size)
    
    return batch_y0, batch_t, batch_y_target




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

    # x_train_mean = torch.tensor(0) 
    # x_train_std = torch.tensor(1) 
    # x_dot_dot_train_mean = torch.tensor(0) 
    # x_dot_dot_train_std = torch.tensor(1) 
   
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

    # Select NN model
    match FUNC_NAME:
        case 'BaselineFNN':
            
            hidden_dim = 32
            params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'activation': ACTIVATION,
                'out_dim': output_dim
            }
            func = BaselineFNN(input_dim, hidden_dim, out_dim=output_dim, activation=activation).to(device)
            wrapped_func = NormalizationWrapper(func, x_dot_mean=x_dot_train_mean, x_dot_std=x_dot_train_std, x_dot_dot_mean=x_dot_dot_train_mean, x_dot_dot_std=x_dot_dot_train_std).to(device)
            
        case 'DeepFNN':
            
            hidden_dim = 256
            n_hidden =  4
            params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'n_hidden': n_hidden,
                'activation': ACTIVATION,
                'out_dim': output_dim
            }

            func = DeepFNN(input_dim, hidden_dim, n_hidden, out_dim=output_dim, activation=activation).to(device)
            wrapped_func = NormalizationWrapper(func, x_dot_mean=x_dot_train_mean, x_dot_std=x_dot_train_std, x_dot_dot_mean=x_dot_dot_train_mean, x_dot_dot_std=x_dot_dot_train_std).to(device)

        case 'LagrangianNN':

            hidden_dim = 512
            params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'activation': ACTIVATION,  
            }
            
            func = LagrangianNN(input_dim=input_dim, hidden_dim=hidden_dim, mean=x_dot_train_mean, std=x_dot_train_std, activation=activation).to(device)
            wrapped_func = func  # LagrangianNN already includes normalization

        case 'StructuredLNN':
            hidden_dim = 256
            params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'activation': ACTIVATION,  
            }
            
            func = StructuredLNN(input_dim=input_dim, hidden_dim=hidden_dim, mean=x_dot_train_mean, std=x_dot_train_std, activation=activation).to(device)
            wrapped_func = func  # LagrangianNN already includes normalization

        case _:
            raise ValueError(f"Unknown FUNC_NAME: {FUNC_NAME}")
        

    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        # 'reduce-overhead' is excellent for small batches/physics loops using CUDA
        # If it crashes, fallback to mode="default"
        wrapped_func = torch.compile(wrapped_func, mode="default")
        
    # ==========================================
    
    

    # select solver and prepare dataloaders if needed
    match SOLVER_NAME:
        case 'ClassicODE':
            # Prepare dataset and dataloader
            dataset_train = TensorDataset(x_train.reshape(-1, 4), x_dot_train.reshape(-1, 4))   # shape: (T*B_train, 4)
            dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
            dataset_test = TensorDataset(x_test.reshape(-1, 4), x_dot_test.reshape(-1, 4))      # shape: (T*B_test, 4)
            dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
            dataset = TensorDataset(x_tensor.reshape(-1, 4), x_dot_tensor.reshape(-1, 4))                     # shape: (T*B, 4)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    
            solver = ClassicODE(wrapped_func).to(device)

            optimizer_parameters = wrapped_func.parameters()

        case 'NeuralODE':
            # No dataloader needed
            solver = NeuralODE(wrapped_func, method='rk4').to(device)
            optimizer_parameters = solver.parameters()
        
     # Select optimizer 
    if OPTIMIZER_NAME == 'AdamW':
        optimizer = optim.AdamW(optimizer_parameters, lr=LEARNING_RATE)
    elif OPTIMIZER_NAME == 'Adam':
        optimizer = optim.Adam(optimizer_parameters, lr=LEARNING_RATE)
    else:
        raise ValueError(f"Unknown OPTIMIZER_NAME: {OPTIMIZER_NAME}")


    # Select Loss
    if LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    elif LOSS =='MAELoss':
        criterion = nn.L1Loss()
    elif LOSS =='HuberLoss':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown LOSS: {LOSS}")
    
    # Training
    print(f"Starting training. \nSolver: {SOLVER_NAME}\nModel: {FUNC_NAME}")

    train_losses = []
    test_losses = []

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=MIN_LR)

    print("---")
    for epoch in range(MAX_EPOCHS+1):

        
        match SOLVER_NAME:
            case 'ClassicODE':
                wrapped_func.train()
                train_loss = train_step(wrapped_func, dataloader_train, device, criterion, optimizer, stats)  
                curricular_step_size = None  # not used in ClassicODE
            case 'NeuralODE':
                solver.train()
                curricular_step_size = curricular_step(epoch, MAX_EPOCHS, initial_chunk_size=2, final_chunk_size=MAX_CHUNK_SIZE)
                batch_y0, batch_t, batch_y_target = get_batch(x_train, T_obs, max_batch_size=BATCH_SIZE, chunk_size=curricular_step_size)
                train_loss = train_evolution(solver, device, batch_y_target, batch_y0, batch_t, criterion, optimizer, stats)

        scheduler.step()
        
        if epoch % TEST_ITER == 0:
            match SOLVER_NAME:
                case 'ClassicODE':
                    test_loss, _ = test_step(wrapped_func, dataloader_test, device, criterion, stats)
                    curricular_step_size = None  # not used in ClassicODE
                case 'NeuralODE':
                    curricular_step_size = curricular_step(epoch, MAX_EPOCHS, initial_chunk_size=2, final_chunk_size=MAX_CHUNK_SIZE)
                    batch_y0, batch_t, batch_y_test = get_batch(x_test, T_obs, max_batch_size=BATCH_SIZE, chunk_size=curricular_step_size)
                    test_loss, _ = test_evolution(solver, device, batch_y_test, batch_y0, batch_t, criterion, stats)

        
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

    # state_dot plots
    match SOLVER_NAME:
        case 'ClassicODE':
            _, x_dot_pred = test_step(wrapped_func, dataloader, device, criterion, stats)
            x_dot_pred = x_dot_pred.reshape(x_dot.shape)

            state_dot_step_plot(x_dot, x_dot_pred, ids=ids, run_dir=run_dir, solver_name=SOLVER_NAME,func_name=FUNC_NAME, frict_type=FRICT_TYPE)

    # Metrics on state evolution
    state_evolution_loss, x_pred = test_evolution(solver, device, x_tensor, x_tensor[0, :, :], T_obs, criterion, stats)
    
    # if FUNC_NAME == 'LagrangianNN' or FUNC_NAME == 'StructuredLNN':
    #     results, energy_pred_model = anomaly_detection(wrapped_func, x_pred)

    #     print("Anomaly Detection Results:")
    #     for key, value in results.items():
    #         print(f"  {key}: {value}")
        
    

    ############################

    x_pred = x_pred.detach().cpu().numpy()

    print(f"State Evolution Loss (MSE): {state_evolution_loss:.5f}")
    state_evolution_plot(x, x_pred, ids=ids, run_dir=run_dir, solver_name=SOLVER_NAME, func_name=FUNC_NAME, frict_type=FRICT_TYPE)


    pendulum = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2, b1=b1, b2=b2)
    energy_pred = compute_energy(pendulum=pendulum, x=x_pred)  # pendulum object not needed here
    energy_evolution_loss = energy_evolution_metric(energy, energy_pred)
    print(f"Energy Evolution Loss ({LOSS}): {energy_evolution_loss:.5f}")

    energy_evolution_plot(energy, energy_pred, ids=ids, run_dir=run_dir, solver_name=SOLVER_NAME, func_name=FUNC_NAME, frict_type=FRICT_TYPE)

    # Save model weights
    torch.save(solver.state_dict(), f"{run_dir}/weights.pth")

    recipe = {
        "solver_name": SOLVER_NAME,
        "func_name": FUNC_NAME,
        "friction_type": FRICT_TYPE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": OPTIMIZER_NAME,
        "loss_function": LOSS,
        "params": params,
        "stats": {
            "x_dot_mean": x_dot_train_mean.cpu().numpy().tolist(),
            "x_dot_std": x_dot_train_std.cpu().numpy().tolist(),
            "x_dot_dot_mean": x_dot_dot_train_mean.cpu().numpy().tolist(),
            "x_dot_dot_std": x_dot_dot_train_std.cpu().numpy().tolist()
        },
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "state_evolution_loss": float(state_evolution_loss),
        "evolution_metric_energy": float(energy_evolution_loss),
        "timestamp": pd.Timestamp.now().isoformat()
    }

    with open(f"{run_dir}/recipe.json", "w") as f:
        json.dump(recipe, f, indent=4)

    history = {
        "epochs": list(range(0, MAX_EPOCHS+1, TEST_ITER)),
        "train_losses": train_losses,
        "test_losses": test_losses
    }

    with open(f"{run_dir}/history.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()