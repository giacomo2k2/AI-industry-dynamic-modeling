# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from NNmodels import BaselineFNN, DeepFNN


# Global constants
WEIGHTS_PATH = 'branch2/weights/'
MODEL_NAME = 'BaselineFNN'  # Options: 'BaselineFNN', 'DeepFNN'
FRICT_TYPE = 'nofrict'

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import csv datasets
    df = pd.read_csv(f"data/processed/double_pendulum_{FRICT_TYPE}.csv")    # Training dataset
    
    time = df['time_step'].unique()
    dt = time[1]-time[0]

    steps = len(time)
    

    # Extract IDs
    x_cols = ['q1', 'q1_dot', 'q2', 'q2_dot']
    y_cols = ['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot']

    weigths_path = WEIGHTS_PATH + MODEL_NAME + '_' + FRICT_TYPE + '.pth'

    match MODEL_NAME:
        case 'BaselineFNN':
            hidden_dim = 32
            model = BaselineFNN(hidden_dim).to(device)

        case 'DeepFNN':
            hidden_dim = 32
            n_hidden =  2

            model = DeepFNN(hidden_dim, n_hidden=n_hidden).to(device)

    print(f"Loading model weights from {weigths_path}...")
    model.load_state_dict(torch.load(weigths_path, map_location=device))

    # 6. Testing Loop
    print("Starting testing...")

    
    model.eval()
    with torch.no_grad():
        states_dot = []

        loss_fn = nn.MSELoss()
        total_loss = 0.0

        for i in range(steps):
            
            x = df[df['step'] == i][x_cols].values.astype(np.float32)
            y = df[df['step'] == i][y_cols].values.astype(np.float32)
            
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
        
            print(f"Time step: {i}/{steps}", end='\r')
            
            
            x_dot = model(x)

            loss = loss_fn(x_dot, y)
            total_loss += loss.item()
            

            states_dot.append(x_dot)

    avg_loss = total_loss / steps
    print(f"\nAverage Test Loss over {steps} steps: {avg_loss:.5f}")

    states_dot = torch.stack(states_dot)  # Shape: (steps, batch_size, state_dim)
    states_dot = states_dot.cpu().detach().numpy()
    
    
    id = 1
    # Visualization of the first pendulum in the batch
    states_dot_gt = df[df['id_pendulum'] == id][y_cols].values.astype(np.float32)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(states_dot[:, id, 0], label='q1_dot')
    ax[0, 0].plot(states_dot_gt[:, 0], label='q1_dot_gt')  
    ax[0, 0].legend()  
    ax[0, 1].plot(states_dot[:, id, 1], label='q1_dot_dot')
    ax[0, 1].plot(states_dot_gt[:, 1], label='q1_dot_dot_gt')
    ax[0, 1].legend()
    ax[1, 0].plot(states_dot[:, id, 2], label='q2_dot')
    ax[1, 0].plot(states_dot_gt[:, 2], label='q2_dot_gt')
    ax[1, 0].legend()
    ax[1, 1].plot(states_dot[:, id, 3], label='q2_dot_dot')
    ax[1, 1].plot(states_dot_gt[:, 3], label='q2_dot_dot_gt')
    ax[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    main()