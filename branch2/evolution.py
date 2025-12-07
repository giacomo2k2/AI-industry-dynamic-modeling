# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from models import DoublePendulum
from NNmodels import BaselineFNN, DeepFNN
from metrics import evolution_metric

from solver import solver, round_to_pi

import json

# Global constants
MODEL_NAME = 'BaselineFNN'  # Options: 'BaselineFNN', 'DeepFNN'
FRICT_TYPE = 'nofrict'
RUN_ID = '001'


def evolution(df, model, device, plot=False, id_plot=0):
     
    time = df['time_step'].unique()
    dt = time[1]-time[0]

    steps = len(time)
    
    m1 = df['m1'][0]
    m2 = df['m2'][0]
    l1 = df['l1'][0]
    l2 = df['l2'][0]
    b1 = df['b1'][0]
    b2 = df['b2'][0]

    pendulum = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2, b1=b1, b2=b2)

    # Extract IDs
    x_cols = ['q1', 'q1_dot', 'q2', 'q2_dot']
    y_cols = ['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot']
    
    model.eval()
    with torch.no_grad():
        states = []

        x = df[df['time_step'] == 0][x_cols].values.astype(np.float32)
       
        x = torch.tensor(x).to(device)
       
        states.append(x)

        for i in range(steps-1):
        
            print(f"Time step: {i}/{steps}", end='\r')
                
            x = solver(x, model, dt=dt, method='fe') # TODO: change rk4
            
            x[:, 0] = round_to_pi(x[:, 0])
            x[:, 2] = round_to_pi(x[:, 2])

            states.append(x)

    states = torch.stack(states)  # Shape: (steps, batch_size, state_dim)
    states = states.cpu().detach().numpy()

    T, B, S = states.shape

    energies = np.zeros((T, B))
    for t in range(T):
        for b in range(B):
            _, _, E = pendulum.energy(states[t, b, :])
            
            energies[t, b] = E

    # create states_gt from df with shape (Time, id, state_dim)
    states_gt = []
    energies_gt = []
    
    for i in range(steps):
        df_t = df[df['step'] == i]
        x_gt = df_t[x_cols].values.astype(np.float32)
        E_gt = df_t['E'].values.astype(np.float32)
        states_gt.append(x_gt)
        energies_gt.append(E_gt)

    states_gt = np.array(states_gt)  # Shape: (steps, batch_size, state_dim)
    energies_gt =  np.array(energies_gt)  # Shape: (steps, batch_size)
    
    metric_state = evolution_metric(states, states_gt)
    metric_energy = evolution_metric(energies, energies_gt)

    id = id_plot
    # Visualization of the first pendulum in the batch
    
    if plot:
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(states[:, id, 0], label='q1')
        ax[0, 0].plot(states_gt[:,id, 0], label='q1_gt')  
        ax[0, 0].legend()  
        ax[0, 1].plot(states[:, id, 1], label='q1_dot')
        ax[0, 1].plot(states_gt[:,id, 1], label='q1_dot_gt')
        ax[0, 1].legend()
        ax[1, 0].plot(states[:, id, 2], label='q2')
        ax[1, 0].plot(states_gt[:,id, 2], label='q2_gt')
        ax[1, 0].legend()
        ax[1, 1].plot(states[:, id, 3], label='q2_dot')
        ax[1, 1].plot(states_gt[:,id, 3], label='q2_dot_gt')
        ax[1, 1].legend()
        plt.xlabel('Time step')
        plt.ylabel('Angle')
        plt.title('Pendulum Angles Over Time')
        plt.show()


        fig = plt.figure()
        plt.plot(energies[:, id], label='Total Energy')
        plt.plot(energies_gt[:, id], label='Total Energy GT')
        plt.xlabel('Time step')
        plt.ylabel('Energy')
        plt.title('Total Energy Over Time')
        plt.legend()
        plt.show()



    return metric_state, metric_energy

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import csv datasets
    df = pd.read_csv(f"data/processed/double_pendulum_{FRICT_TYPE}.csv")    # Training dataset
    
    weigths_path = f"branch2/runs/{MODEL_NAME}_{FRICT_TYPE}_{RUN_ID}/weights.pth"
    
    # import params from json recipe.json
    with open(f"branch2/runs/{MODEL_NAME}_{FRICT_TYPE}_{RUN_ID}/recipe.json", 'r') as f:
        recipe = json.load(f)
    params = recipe['params']

    match MODEL_NAME:
        case 'BaselineFNN':
            model = BaselineFNN(params).to(device)

        case 'DeepFNN':
            model = DeepFNN(params).to(device)

    print(f"Loading model weights from {weigths_path}...")
    model.load_state_dict(torch.load(weigths_path, map_location=device))

    
    metric_state, metric_energy = evolution(df, model, device, plot=True, id_plot=0)

    print(f"\nState Evolution Metric: {metric_state:.5f}")
    print(f"Energy Evolution Metric: {metric_energy:.5f}")


if __name__ == "__main__":
    main()