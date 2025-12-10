# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from NNmodels import BaselineFNN, DeepFNN, LagrangianNN, ClassicODE, NeuralODE
from models import DoublePendulum

import os
import json

from utils import generate_run_dir, history_plot, state_dot_step_plot, state_evolution_plot, compute_energy, energy_evolution_plot, round_to_pi
from metrics import energy_evolution_metric

# Global constants
MAX_EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEST_ITER = 10
TRAIN_TEST_SPLIT = 0.8

ACTIVATION = 'SiLU'  # Options: 'ReLU', 'Tanh', 'SiLU', 'GELU', SELU, 'Softplus'

SOLVER_NAME = 'ClassicODE'  # Options: 'ClassicODE', 'NeuralODE'
FUNC_NAME = 'LagrangianNN'  # Options: 'BaselineFNN', 'DeepFNN', 'HamiltonianNN', 'LagrangianNN'

FRICT_TYPE = 'nofrict'
OPTIMIZER_NAME = 'AdamW' # Options: 'AdamW', 'Adam'
LOSS = 'MSELoss'


def train_step(func, dataloader, device, criterion, optimizer):
    # Train a single step ahead using the batchsize in the dataloader
    # Dataset: shuffled    
    running_loss = 0.0
    for x_batch, x_dot_dot_batch in dataloader:
        x_batch, x_dot_dot_batch = x_batch.to(device), x_dot_dot_batch.to(device)
        
        optimizer.zero_grad()                       # Reset gradients
        x_dot_pred = func(t=None, x=x_batch)                  # Forward pass
    
        loss = criterion(x_dot_pred[:, 2:], x_dot_dot_batch)   # Loss
        loss.backward()                             # Backpropagation
        optimizer.step()                            # Update weights      
        running_loss += loss.item()

    return running_loss / len(dataloader)

def train_evolution(solver, device, x, x0, T, criterion, optimizer):
    # Train the model to learn all the future evolution of the model
    # Model: call the solver + func
    # Use-case: Neural ODE 
    x0 = x0.to(device)
    x = x.to(device)
    T = T.to(device)

    optimizer.zero_grad()                       # Reset gradients
    
    x_pred = solver(x0, T)                       # Forward pass
    train_loss = criterion(x_pred, x)           # Loss
    
    train_loss.backward()                       # Backpropagation
    
    optimizer.step()                            # Update weights        

    return train_loss.item()
        
                        
def test_step(func, dataloader, device, criterion):
    # Test the single step ahead training
    # Out: derivative of the next step. THen a solver integration is necessary
    func.eval()
    total_loss = 0.0

    predictions = []
    with torch.no_grad():
        for x_batch, x_dot_dot_batch in dataloader:
            x_batch, x_dot_dot_batch = x_batch.to(device), x_dot_dot_batch.to(device)

            x_dot_pred = func(t=None, x=x_batch)
            
            predictions.append(x_dot_pred.detach().cpu().numpy())

            loss = criterion(x_dot_pred[:, 2:], x_dot_dot_batch)
            total_loss += loss.item()

    predictions = np.concatenate(predictions, axis=0)
    return total_loss / len(dataloader), predictions


def test_evolution(solver, device, x, x0, T, criterion):
    # Test the all time evolution
    # Out: all time step states already integred
    solver.eval()
    with torch.no_grad():
        x0 = x0.to(device)
        x = x.to(device)
        T = T.to(device)

        x_pred = solver(x0, T)
        test_loss = criterion(x_pred, x)
    return test_loss.item(), x_pred.detach().cpu().numpy()


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
    x_dot_dot_cols = ['q1_dot_dot', 'q2_dot_dot']
    x_dot_cols = ['q1_dot', 'q2_dot', 'q1_dot_dot', 'q2_dot_dot']
    # Extract x and x_dot (shape: (B*T, 4))
    x = []
    x_dot = []
    x_dot_dot = []
    energy = []

    for step in df['step'].unique():
        df_step = df[df['step'] == step]
        x_step = df_step[x_cols].values.astype(np.float32)
        x_dot_step = df_step[x_dot_cols].values.astype(np.float32)
        x_dot_dot_step = df_step[x_dot_dot_cols].values.astype(np.float32)
        energy_step = df_step['E'].values.astype(np.float32)
        x.append(x_step)
        x_dot.append(x_dot_step)
        x_dot_dot.append(x_dot_dot_step)
        energy.append(energy_step)

    # stack along time dimension 
    x = np.stack(x)             # Shape: (T, B, 4)
    x_dot = np.stack(x_dot)         # Shape: (T, B, 4)
    x_dot_dot = np.stack(x_dot_dot)     # Shape: (T, B, 2)
    energy = np.stack(energy)   # Shape: (T, B)

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)             # Shape: (T, B, 4)
    x_dot_dot_tensor = torch.tensor(x_dot_dot, dtype=torch.float32)     # Shape: (T, B, 2)

    # Time observations tensor
    T_obs = torch.tensor(df['time_step'].unique(), dtype=torch.float32)  # Shape: (T,)

    # Train-test split
    num_pendulums = x_tensor.shape[1]
    split_index = int(num_pendulums * TRAIN_TEST_SPLIT)

    x_train = x_tensor[:, :split_index, :]          # shape: (T, B_train, 4)
    x_dot_dot_train = x_dot_dot_tensor[:, :split_index, :]  # shape: (T, B_train, 2)
 
    x_test = x_tensor[:, split_index:, :]           # shape: (T, B_test, 4)
    x_dot_dot_test = x_dot_dot_tensor[:, split_index:, :]   # shape: (T, B_test, 2)

    input_dim = len(x_cols)  # 4
    output_dim = len(x_dot_dot_cols)  # 2

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

        case 'LagrangianNN':

            hidden_dim = 512
            params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'activation': ACTIVATION,  
            }
            
            func = LagrangianNN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

        case _:
            raise ValueError(f"Unknown FUNC_NAME: {FUNC_NAME}")
        
    
    # select solver and prepare dataloaders if needed
    match SOLVER_NAME:
        case 'ClassicODE':
            # Prepare dataset and dataloader
            dataset_train = TensorDataset(x_train.reshape(-1, 4), x_dot_dot_train.reshape(-1, 2))   # shape: (T*B_train, 4)
            dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
            dataset_test = TensorDataset(x_test.reshape(-1, 4), x_dot_dot_test.reshape(-1, 2))      # shape: (T*B_test, 4)
            dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
            dataset = TensorDataset(x_tensor.reshape(-1, 4), x_dot_dot_tensor.reshape(-1, 2))       # shape: (T*B, 4)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            solver = ClassicODE(func).to(device)

            optimizer_parameters = func.parameters()

        case 'NeuralODE':
            # No dataloader needed
            solver = NeuralODE(func, method='dopri5').to(device)

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
    else:
        raise ValueError(f"Unknown LOSS: {LOSS}")
    
    # Training
    print(f"Starting training. \nSolver: {SOLVER_NAME}\nModel: {FUNC_NAME}")

    train_losses = []
    test_losses = []

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    print("---")
    for epoch in range(MAX_EPOCHS+1):

        
        match SOLVER_NAME:
            case 'ClassicODE':
                func.train()
                train_loss = train_step(func, dataloader_train, device, criterion, optimizer)  
            case 'NeuralODE':
                solver.train()
                train_loss = train_evolution(solver, device, x_train, x_train[0, :, :], T_obs, criterion, optimizer)
        scheduler.step()
        
        if epoch % TEST_ITER == 0:
            match SOLVER_NAME:
                case 'ClassicODE':
                    test_loss, _ = test_step(func, dataloader_test, device, criterion)
                case 'NeuralODE':
                    test_loss, _ = test_evolution(solver, device, x_test, x_test[0, :, :], T_obs, criterion)

        
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch: {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")
        
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
            _, x_dot_pred = test_step(func, dataloader, device, criterion)
            x_dot_pred = x_dot_pred.reshape(x_dot.shape)

            
            state_dot_step_plot(x_dot, x_dot_pred, ids=ids, run_dir=run_dir, solver_name=SOLVER_NAME,func_name=FUNC_NAME, frict_type=FRICT_TYPE)

    # Metrics on state evolution
    state_evolution_loss, x_pred = test_evolution(solver, device, x_tensor, x_tensor[0, :, :], T_obs, criterion)
    x_pred[:, :, :2] = round_to_pi(x_pred[:, :, :2])  # wrap angles to [-pi, pi]
    
    print(f"State Evolution Loss (MSE): {state_evolution_loss:.5f}")
    state_evolution_plot(x, x_pred, ids=ids, run_dir=run_dir, solver_name=SOLVER_NAME, func_name=FUNC_NAME, frict_type=FRICT_TYPE)



    pendulum = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2, b1=b1, b2=b2)
    energy_pred = compute_energy(pendulum=pendulum, x=x_pred)  # pendulum object not needed here
    energy_evolution_loss = energy_evolution_metric(energy, energy_pred)
    print(f"Energy Evolution Loss (MSE): {energy_evolution_loss:.5f}")

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