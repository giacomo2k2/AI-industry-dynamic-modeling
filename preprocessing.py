import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from model import DoublePendulum
from simulation import Simulator
from simulation import Simulator

def process_data(path:str):

    df = pd.read_csv(f'data/raw/{path}')

    processed_df = pd.DataFrame(columns=df.columns)

    for id in df['id_pendulum'].unique():

        #select a pendulum
        pendulum_id = id
        pendulum_data = df[df['id_pendulum'] == pendulum_id]

        #create a numpy array (steps, 4) with the o1, o1_dot, o2, o2_dot
        states = pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()

        total_energies = []

        m1 = pendulum_data['m1'].iloc[0]
        m2 = pendulum_data['m2'].iloc[0]
        l1 = pendulum_data['l1'].iloc[0]
        l2 = pendulum_data['l2'].iloc[0]
        b1 = pendulum_data['b1'].iloc[0]
        b2 = pendulum_data['b2'].iloc[0]

        doublePendulum = DoublePendulum(states[0, :], m1, m2, l1, l2, b1, b2)
        # for each state compute total energy
        for i in range(states.shape[0]):
            o = (states[i,0], states[i,2])
            o_dot = (states[i,1], states[i,3])

            _, _, E = doublePendulum.energy(o, o_dot)
            total_energies.append(E)

            if i>0 and total_energies[i] > total_energies[i-1] + 1e-2: # bad pendulum
                print(f"Energy not conserved for pendulum id {pendulum_id} at step {i}: {total_energies[i-1]} -> {total_energies[i]}")
                break
        
        
        if i>0 and total_energies[i] > total_energies[i-1] + 1e-2: # bad pendulum
            continue
        else:
            processed_df = pd.concat([processed_df, pendulum_data], ignore_index=True)
        
    # save processed dataframe to csv
    processed_df.to_csv(f'data/processed/{path}', index=False)

    # print process terminated and number of pendulums kept
    n_total = df['id_pendulum'].nunique()
    n_kept = processed_df['id_pendulum'].nunique()
    print(f"Processing terminated. Kept {n_kept} out of {n_total} pendulums.")


def show_example(path:str, id:int = 0):

    simulator = Simulator()

    df = pd.read_csv(path)

    #select a pendulum
    pendulum_id = id
    pendulum_data = df[df['id_pendulum'] == pendulum_id]

    #create a numpy array (steps, 4) with the o1, o1_dot, o2, o2_dot
    states = pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()

    #extract dt
    dt = pendulum_data['time_step'].iloc[1] - pendulum_data['time_step'].iloc[0]

    #extract masses
    m1 = pendulum_data['m1'].iloc[0]
    m2 = pendulum_data['m2'].iloc[0]

    #extract lengths
    l1 = pendulum_data['l1'].iloc[0]
    l2 = pendulum_data['l2'].iloc[0]

    # extract friction
    b1 = pendulum_data['b1'].iloc[0]
    b2 = pendulum_data['b2'].iloc[0]

    # create pendulum
    doublePendulum = DoublePendulum(states[0, :], m1, m2, l1, l2, b1, b2)

    #plot the energies
    doublePendulum.energy_plot(states, dt)

    # animation
    simulator.draw_simulation(doublePendulum, states)


if __name__ == '__main__':

    show_example('data/processed/double_pendulum_fixed.csv', id=10)

    #process_data('double_pendulum_data.csv')