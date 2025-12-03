import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from model import DoublePendulum
from simulation import Simulator
from simulation import Simulator

def process_data(path:str):

    # Read the csv in the data raw folder
    df = pd.read_csv(f'data/raw/{path}')

    processed_df = pd.DataFrame(columns=df.columns)

    for id in df['id_pendulum'].unique():

        # Select a pendulum
        pendulum_id = id

        # Extract pendulum data from DataFrame
        pendulum_data = df[df['id_pendulum'] == pendulum_id]

        # Create a numpy array (steps, 4) with the o1, o1_dot, o2, o2_dot
        states = pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()

        # Time evolution of mechanical energy 
        total_energies = []

        # Extract pendulum params
        m1 = pendulum_data['m1'].iloc[0]
        m2 = pendulum_data['m2'].iloc[0]
        l1 = pendulum_data['l1'].iloc[0]
        l2 = pendulum_data['l2'].iloc[0]
        b1 = pendulum_data['b1'].iloc[0]
        b2 = pendulum_data['b2'].iloc[0]

        # Create a DoublePedulum object with given params
        doublePendulum = DoublePendulum(states[0, :], m1, m2, l1, l2, b1, b2)

        # For each state compute total energy
        for i in range(states.shape[0]):
            o = (states[i,0], states[i,2])
            o_dot = (states[i,1], states[i,3])

            _, _, E = doublePendulum.energy(o, o_dot)
            total_energies.append(E)

            # Check if a pendolum does not conserve the energy
            if i>0 and total_energies[i] > total_energies[i-1] + 1e-2: 
                print(f"Energy not conserved for pendulum id {pendulum_id} at step {i}: {total_energies[i-1]} -> {total_energies[i]}")
                break
        
        # If a pendolum does not conserve the energy skip concatenation
        if i>0 and total_energies[i] > total_energies[i-1] + 1e-2: 
            continue
        else:
            # Concatenate the pedolum data over the processed dataframe
            processed_df = pd.concat([processed_df, pendulum_data], ignore_index=True)
            
        processed_df = round_to_pi(processed_df)
        
    # Save processed dataframe to csv
    processed_df.to_csv(f'data/processed/{path}', index=False)

    # Print process terminated and number of pendulums kept
    n_total = df['id_pendulum'].nunique()
    n_kept = processed_df['id_pendulum'].nunique()
    print(f"Processing terminated. Kept {n_kept} out of {n_total} pendulums.")


def show_example(path:str, id:int = 0):
    """
    Plot the energies and draw the animation for a given pendulum from the csv file.
    """
    simulator = Simulator()
    
    # Read the csv
    df = pd.read_csv(path)

    # Select a pendulum
    pendulum_id = id

    # Extract pendulum data from DataFrame
    pendulum_data = df[df['id_pendulum'] == pendulum_id]

    # Create a numpy array (steps, 4) with the o1, o1_dot, o2, o2_dot
    states = pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()

    # Extract dt from DataFrame
    dt = pendulum_data['time_step'].iloc[1] - pendulum_data['time_step'].iloc[0]

    # Extract pendulum params
    m1 = pendulum_data['m1'].iloc[0]
    m2 = pendulum_data['m2'].iloc[0]
    l1 = pendulum_data['l1'].iloc[0]
    l2 = pendulum_data['l2'].iloc[0]
    b1 = pendulum_data['b1'].iloc[0]
    b2 = pendulum_data['b2'].iloc[0]

    # Create a DoublePedulum object with given params
    doublePendulum = DoublePendulum(states[0, :], m1, m2, l1, l2, b1, b2)

    # Plot the energies
    doublePendulum.energy_plot(states, dt)

    # Draw the animation
    simulator.draw_simulation(doublePendulum, states)

def compute_statistics(path:str, id:int = 0):
    """
    Compute the expected value and standard deviation for o1, o2, o1_dot, o2_dot
    """

    # Read the csv
    df = pd.read_csv(path)

    # Select a pendulum
    pendulum_id = id

    # Extract pendulum data from DataFrame
    pendulum_data = df[df['id_pendulum'] == pendulum_id]

    # Create a numpy array (steps, 4) with the o1, o1_dot, o2, o2_dot
    states = pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()

    # Compute expected value and standard deviation of states 
    means = np.mean(states, axis=0)
    stds = np.std(states, axis=0)

    # Show results
    print(f"E[ o1 ] = {means[0]}, Std( o1 ) = {stds[0]}")
    print(f"E[ o2 ] = {means[1]}, Std( o2 ) = {stds[1]}")
    print(f"E[ o1_dot ] = {means[2]}, Std( o1_dot ) = {stds[2]}")
    print(f"E[ o2_dot ] = {means[3]}, Std( o2_dot ) = {stds[3]}")

def show_bins(path:str, id:int = 0):

    # Plot the histogram for o1, o2, o1_dot, o2_dot
    df = pd.read_csv(path)

    if (id != 0):
        #select a pendulum
        pendulum_id = id
        pendulum_data = df[df['id_pendulum'] == pendulum_id]
    else:
        pendulum_data = df

    # extract states
    states = pendulum_data[['id_pendulum', 'o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()

    #DEBUG
    #pendulum_count = (int)(states[-1, 0]+1)
    #initial_conditions = np.zeros((pendulum_count, 4))
    #for i in range(pendulum_count):
    #    initial_conditions[i, :] = states[states[:,0]==i][0,1:5]
    #print(pendulum_data.shape)


    # bin1 : bin of angles for o1
    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    n, bins, patches = plt.hist(states[:,1], bins=100, range=(-np.pi, np.pi), density=True)
    plt.title("Histogram of o1")
    plt.xlabel("o1 [rad]")
    plt.ylabel("Density")
    # bin2 : bin of angular velocities for o1_dot
    plt.subplot(2,2,2)
    n, bins, patches = plt.hist(states[:,2], bins=100, range=(-15, 15), density=True)
    plt.title("Histogram of o1_dot")
    plt.xlabel("o1_dot [rad/s]")
    plt.ylabel("Density")
    # bin1 : bin of angles for o2
    plt.subplot(2,2,3)
    n, bins, patches = plt.hist(states[:,3], bins=100, range=(-np.pi, np.pi), density=True)
    plt.title("Histogram of o2")
    plt.xlabel("o2 [rad]")
    plt.ylabel("Density")
    # bin2 : bin of angular velocities for o2_dot
    plt.subplot(2,2,4)
    n, bins, patches = plt.hist(states[:,4], bins=100, range=(-15, 15), density=True)
    plt.title("Histogram of o2_dot")
    plt.xlabel("o2_dot [rad/s]")
    plt.ylabel("Density")
    plt.show()


def round_to_pi(df: pd.DataFrame):
    """
    Wrap angles to the interval [-pi, pi]
    """
    # ATTENTION: watchout the module, in NN it may not work.
    o = df[['o1', 'o2']].to_numpy() + np.pi
    o = np.mod(o, 2*np.pi) - np.pi
    df['o1'], df['o2'] = o[:,0], o[:,1]

    return df

    

if __name__ == '__main__':

    example_id = 0
    
    #compute_statistics('data/processed/double_pendulum_fixed.csv', id=example_id)

    #show_example('data/raw/double_pendulum_nofrict.csv', id=example_id)
    #show_example('data/raw/double_pendulum_lowfrict.csv', id=example_id)
    #show_example('data/raw/double_pendulum_highfrict.csv', id=example_id)

    process_data('double_pendulum_highfrict.csv')
    show_bins('data/processed/double_pendulum_highfrict.csv', id=example_id)