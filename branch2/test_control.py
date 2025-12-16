# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

import torch.optim as optim

from NNmodels import BaselineFNN, DeepFNN, LagrangianNN, ClassicODE, NeuralODE, NormalizationWrapper, StructuredLNN
from models import DoublePendulum

from simulation import Simulator


import os 


ACTIVATION = 'SiLU'  # Options: 'ReLU', 'Tanh', 'SiLU', 'GELU', SELU, 'Softplus'

SOLVER_NAME = 'ClassicODE'  # Options: 'ClassicODE', 'NeuralODE'
FUNC_NAME = 'StructuredLNN'  # Options: 'BaselineFNN', 'DeepFNN', 'HamiltonianNN', 'LagrangianNN'

FRICT_TYPE = 'nofrict'

RUNID = '015'  # Options: 'latest' or specific run folder name

TRAIN_TEST_SPLIT = 0.8

def test_evolution(solver, device, x0, T,):
    # Test the all time evolution
    # Out: all time step states already integred

    solver.eval()
    x0 = x0.to(device)
    
    T = T.to(device)
    x_pred = solver(x0, T)

    return x_pred

def main():
    # torch seed
    torch.manual_seed(42)

    # Set device    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import full csv dataset (shape: (B*T, C))
    df = pd.read_csv(f"data/processed/double_pendulum_nofrict.csv")  
    
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

    input_dim = 4
    output_dim = 2

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
            
            func = StructuredLNN(input_dim=input_dim, hidden_dim=hidden_dim, mean=x_dot_train_mean, std=x_dot_train_std, activation=activation, control=True, kgain=0.2).to(device)
            wrapped_func = func  # LagrangianNN already includes normalization


        case _:
            raise ValueError(f"Unknown FUNC_NAME: {FUNC_NAME}")
    

    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        # 'reduce-overhead' is excellent for small batches/physics loops using CUDA
        # If it crashes, fallback to mode="default"
        wrapped_func = torch.compile(wrapped_func, mode="default")
        

    # select solver and prepare dataloaders if needed
    match SOLVER_NAME:
        case 'ClassicODE':    
            solver = ClassicODE(wrapped_func).to(device)

        
        case 'NeuralODE':
            # No dataloader needed
            solver = NeuralODE(wrapped_func, method='rk4').to(device)
            
    # extract model weights from runs/solver_name/func_name/frict_type/runid
    if RUNID == 'latest':
        base_path = f"branch2/runs/{SOLVER_NAME}_{FUNC_NAME}_{FRICT_TYPE}/"
        run_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        run_folders.sort(reverse=True)  # Sort to have latest first
        run_folder = run_folders[0]  # Get the latest run folder
    else:
        run_folder = RUNID

    weights_path = f"branch2/runs/{SOLVER_NAME}_{FUNC_NAME}_{FRICT_TYPE}_{run_folder}/weights.pth"

    solver.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded model weights from {weights_path}")
    
    
    
    tend = 30.0
    dt = 0.01

    T = np.arange(0, tend + dt, dt).astype(np.float32)  # Shape: (T,)
    T = torch.tensor(T, dtype=torch.float32)
    
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Initial state
    x0 = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)

    # Metrics on state evolution
    x_pred = test_evolution(solver, device, x0, T)
    
    x_pred = x_pred.detach().cpu().numpy()

    #plot
    pendulum = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2, b1=b1, b2=b2)
    simulator = Simulator(dt=dt)

    simulator.draw_simulation(model=pendulum, states=x_pred[:, 0, :], target_fps=30)

    

if __name__ == "__main__":
    main()