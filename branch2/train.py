# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Global constants
MAX_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

TEST_ITER = 1

EVAL_ITER = 10

HIDDEN_DIM = 32


# Pytorch FF NN
class feedforward_nn_model(nn.Module):
    def __init__(self, input_dim, internal_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, internal_dim)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(internal_dim, internal_dim)
        self.output = nn.Linear(internal_dim, output_dim)
        
    def forward(self, x):
        x = self.input(x)           # B, C
        x = self.activation(x)      # B, C
        x = self.linear1(x)         # B, C
        x = self.activation(x)      # B, C    
        x = self.output(x)          # B, C

        return x

def get_batch(dataloader, device):
    x = []
    y = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x.append(x_batch.to(device))
            y.append(y_batch.to(device))

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    return x, y

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



def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import csv datasets
    df_train = pd.read_csv("data/training/train_nofrict.csv")    # Training dataset
    df_test = pd.read_csv("data/training/test_nofrict.csv")      # Test dataset
    
    x_cols = ['q1', 'q1_dot', 'q2', 'q2_dot']
    y_cols = ['q1_dot_dot', 'q2_dot_dot']

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

    in_dim = len(x_cols)
    out_dim = len(y_cols)

    # MLP model
    model = feedforward_nn_model(in_dim, HIDDEN_DIM, out_dim).to(device)

    # Compile model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print("Starting training...")

    train_losses = []
    test_losses = []

    # train
    for epoch in range(MAX_EPOCHS+1):
        model.train()
        
        running_loss = 0.0
        
        # Mini-batch Gradient Descent
        for x_batch, y_batch in dataloader_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()           # Reset gradients
            y_pred = model(x_batch)         # Forward
        
            loss = criterion(y_pred, y_batch) # Loss
            loss.backward()                 # Backward
            optimizer.step()                # Update weights
        
    
            running_loss += loss.item()

        train_loss = running_loss / len(dataloader_train)
        
        
        if epoch % TEST_ITER == 0:
            test_loss = eval_model(model, dataloader_test, device, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch: {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")

    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()