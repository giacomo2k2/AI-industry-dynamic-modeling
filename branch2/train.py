# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from evolution import evolution
from NNmodels import BaselineFNN, DeepFNN

import json

import os
from datetime import datetime
import re

# Global constants
MAX_EPOCHS = 500
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
TEST_ITER = 10
MODEL_NAME = 'DeepFNN'  # Options: 'BaselineFNN', 'DeepFNN'
FRICT_TYPE = 'nofrict'
OPTIMIZER_NAME = 'AdamW'
LOSS = 'MSELoss'


def eval_model(model, dataloader, device, criterion):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculation
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            num_batches += 1

    model.train() # Reset to train mode
    return total_loss / num_batches




def generate_run_dir(base_dir, model_name, frict_type):
    pattern = re.compile(rf"{model_name}_{frict_type}_(\d+)$")

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
    run_name = f"{model_name}_{frict_type}_{new_id:03d}"
    run_dir = os.path.join(base_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir, new_id



def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import csv datasets
    df_train = pd.read_csv(f"data/training/train_{FRICT_TYPE}.csv")    # Training dataset
    df_test = pd.read_csv(f"data/training/test_{FRICT_TYPE}.csv")      # Test dataset
    
    x_cols = ['q1', 'q1_dot', 'q2', 'q2_dot']
    y_cols = ['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot']

    # Extract input and output
    x_train = df_train[x_cols].values.astype(np.float32)
    y_train = df_train[y_cols].values.astype(np.float32)
    x_test = df_test[x_cols].values.astype(np.float32)
    y_test = df_test[y_cols].values.astype(np.float32) 
    
    # TensorDataset
    dataset_train = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    dataset_test = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    
    # Dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # Slect NN model
    match MODEL_NAME:
        case 'BaselineFNN':
            hidden_dim = 32
            params = {
                'hidden_dim': hidden_dim
            }
            model = BaselineFNN(params).to(device)

        case 'DeepFNN':
            hidden_dim = 256
            n_hidden =  8
            params = {
                'hidden_dim': hidden_dim,
                'n_hidden': n_hidden
            }

            model = DeepFNN(params).to(device)
    
        case _:
            raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")
        
    # Slect loss
    if LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown LOSS: {LOSS}")
    
    # Slect optimizer 
    if OPTIMIZER_NAME == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError(f"Unknown OPTIMIZER_NAME: {OPTIMIZER_NAME}")

    # Training
    print(f"Starting training {MODEL_NAME} model...")

    train_losses = []
    test_losses = []

    for epoch in range(MAX_EPOCHS+1):
        model.train()
        
        running_loss = 0.0
        
        # Mini-batch Gradient Descent
        for x_batch, y_batch in dataloader_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()                   # Reset gradients
            y_pred = model(x_batch)                 # Forward pass
        
            loss = criterion(y_pred, y_batch)       # Loss
            loss.backward()                         # Backpropagation
            optimizer.step()                        # Update weights
    
            running_loss += loss.item()

        train_loss = running_loss / len(dataloader_train)
        
        # Verbose every TEST_ITER  
        if epoch % TEST_ITER == 0:
            test_loss = eval_model(model, dataloader_test, device, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch: {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")

    # History plots
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()


    df_processed = pd.read_csv(f"data/processed/double_pendulum_{FRICT_TYPE}.csv")    # Training dataset
    metric_evolution_state, metric_evolution_energy = evolution(df_processed, model, device, plot=False, id_plot=0)

    # Save model config in runs folder
    run_dir, run_id = generate_run_dir('branch2/runs/', MODEL_NAME, FRICT_TYPE)
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{run_dir}/weights.pth")

    recipe = {
        "model_name": MODEL_NAME,
        "friction_type": FRICT_TYPE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": OPTIMIZER_NAME,
        "loss_function": LOSS,
        "params": params,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "evolution_metric_state": float(metric_evolution_state),
        "evolution_metric_energy": float(metric_evolution_energy),
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