import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from simulation import Simulator
from models import DoublePendulum

plt.rc('font', family='serif')

"""
This module is responsible for visualizing the state-space evolution of 
mechanical systems (specifically double pendulums) across multiple simulation 
instances. It converts tabular data into heatmap visualizations to easily 
identify patterns, stability, or anomalies in the dataset.
"""

# ========================================================
# ================== SHOW THE DATASET ==================
# ========================================================


def show_status_stats(df: pd.DataFrame, cols: list, names:list, filename  : str = '', save: bool = False):
    """
    Compute variance of each state across all pendulums for each time step.

    ---
    Args:
        df (pd.DataFrame): DataFrame containing the simulation data.
        cols (list): List of column names representing the state variables.
        names (list): List of human-readable names for the state variables.
        filename (str, optional): Filename to save the plot. Defaults to ''.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """
    
    time_steps = df['time_step'].unique()
    stds = np.zeros((len(time_steps), len(cols)))
    means = np.zeros((len(time_steps), len(cols)))
    
    
    for i, t in enumerate(time_steps):
        step_data = df[df['time_step'] == t]
        states = step_data[cols].to_numpy()
        stds[i, :] = np.std(states, axis=0)
        means[i, :] = np.mean(states, axis=0)
       
    
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    for (i, a) in enumerate(ax.flatten()):
        a.plot(stds[:, i])
        a.plot(means[:, i], color='orange')
        a.grid()
        a.legend([r'$\sigma$', r'$\mu$'])
        a.set_title(f'{names[i]}')

    plt.tight_layout()
    plt.show()

    if save:
        # save the plot as png
        fig.savefig('plots/variance_means_' + filename +'.png')

def show_energy_stats(df:pd.DataFrame, filename:str='', save:bool = False):
    """
    Keep onlty the time steps lower than a threshold.

    ----
    Args:
        df (pd.DataFrame): DataFrame containing the simulation data.
        filename (str, optional): Filename to save the plot. Defaults to ''.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """

    # Compute for each time step the medium total energy
    time_steps = df['time_step'].unique()
    
    std_energies = []
    mean_energies = []


    for t in time_steps:
        step_data = df[df['time_step'] == t]
        energy_values = step_data['E'].to_numpy()
        mean_energy = np.mean(energy_values)
        std_energy = np.std(energy_values)        

        mean_energies.append(mean_energy)
        std_energies.append(std_energy)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(std_energies)
    plt.plot(mean_energies, color='orange')
    plt.title('Energies mean and std')
    plt.grid()
    plt.legend([r'$\sigma$', r'$\mu$'])
    plt.tight_layout()
    plt.show()

    if save:
    #save the plot as png
        fig.savefig('plots/energy_stats_' + filename + '.png')
    

def show_bins(df: pd.DataFrame,  cols:list, names: list, vlim: np.ndarray, save: bool = False, filename: str = ''):
    


    pendulum_data = df

    # extract states
    states = pendulum_data[cols].to_numpy()


    fig, axes = plt.subplots(2, 3, figsize=(12, 6))


    configs = [
        (states[:, 0], (-vlim[0], vlim[0]),    names[0]),
        (states[:, 1], (-vlim[1], vlim[1]),    names[1]),
        (states[:, 2], (-vlim[2], vlim[2]),    names[2]),
        (states[:, 3], (-vlim[3], vlim[3]),    names[3]),
        (states[:, 4], (-vlim[4], vlim[4]),    names[4]),
        (states[:, 5], (-vlim[5], vlim[5]),    names[5]),
    ]


    for ax, (data, rng, title) in zip(axes.ravel(), configs):
        ax.hist(data, bins=100, range=rng, density=True)
        ax.set_title(f"Histogram of {title}")
        ax.set_ylabel("Density")

    fig.tight_layout()
    plt.show()


    if save and id == -1:
        # save the plot as png
        fig.savefig('plots/histogram_pendulum_' + filename + '.png')



def show_heatmap(df, cols: list, names : list, vlim: np.ndarray, pendulum_id: str ='id_pendulum', time: str ='time_step',
                      figsize: tuple =(12,6), file_name:str = None) -> None:
    """
    Generates a 2x2 grid of heatmaps visualizing the evolution of state variables over time.

    This function pivots the input DataFrame to organize data by simulation ID 
    and time, then plots the magnitude of the specified state variables (inputs) 
    using a color map.

    ---
    Args:
        df (pd.DataFrame): The source dataframe containing simulation data. 
                           Must contain columns for ID, time, and the state variables.
        inputs (list of str): A list of column names representing the state variables 
                              to plot (e.g., ['q1', 'q2', 'p1', 'p2']). 
                              Note: This function is optimized for 4 variables.
        pendulum_id (str, optional): The column name identifying unique pendulum 
                                     simulations/trajectories. Defaults to 'id_pendulum'.
        time (str, optional): The column name representing the time step. 
        vlim (np.ndarray): Limit for position, velocity and acceleration.
        figsize (tuple, optional): The dimensions of the generated figure (width, height). 
        file_name (str, optional): Name necessary to save the subplot image.

    ---
    Returns:
        None: Displays the plot window via plt.show().
    """
    # Initialize a 2x2 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    fig.suptitle('State Variables Heatmaps Over Time', fontsize=16)
    
    # Zip flattens the 2D axes array to iterate linearly along with the input variables
    for ax, col, name, abs_vlim in zip(axes.flatten(), cols, names, vlim):
        # Determine vmin and vlim for color scaling
        
            
        # Pivot table: Rows = Simulation ID, Cols = Time, Values = State Variable
        pivot = df.pivot(index=pendulum_id, columns=time, values=col)

        # Plot heatmap
        # aspect='auto' ensures the pixels fill the plot area regardless of data shape
        im = ax.imshow(pivot, aspect='auto', cmap='RdBu', vmin=-abs_vlim, vmax=abs_vlim)

        # Decoration
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pendulum ID")
        
    plt.tight_layout()
    plt.show()
    
    # Save the image as a png file
    if not file_name:
        file_name = 'subplots'+'_'.join(cols)
    fig.savefig(file_name+".png")

def show_example(path:str, id:int = 0, animate: bool = False):
    """ T = 1
    get_variance(df, T, True)
    Plot the energies and draw the animation for a given pendulum from the csv file.
    """
    df = pd.read_csv(path)

    pendulum_data = df[df['id_pendulum'] == id]

    time = pendulum_data['time_step'].to_numpy()
    T = pendulum_data['T'].to_numpy()
    U = pendulum_data['U'].to_numpy()
    E = pendulum_data['E'].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(time, T, label='Kinetic Energy')
    plt.plot(time, U, label='Potential Energy')
    plt.plot(time, E, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    if animate:
        dt = time[1] - time[0]
        simulator = Simulator(dt)

        pendulum = DoublePendulum(
                m1 = df['m1'].iloc[0],
                m2 = df['m2'].iloc[0],
                l1 = df['l1'].iloc[0],
                l2 = df['l2'].iloc[0],
                b1 = df['b1'].iloc[0],
                b2 = df['b2'].iloc[0],
        )

        simulator.draw_simulation(pendulum, pendulum_data[['q1', 'q1_dot', 'q2', 'q2_dot']].to_numpy())

def main():
    example_id = 55
    # Define the states
    cols = ['q1', 'q1_dot','q1_dot_dot', 'q2', 'q2_dot', 'q2_dot_dot']
    status_names = [r'$q_1$', r'$\dot{q}_1$', r'$\ddot{q}_1$', r'$q_2$', r'$\dot{q}_2$',  r'$\ddot{q}_2$']

    files = [
        'double_pendulum_nofrict.csv',
        'double_pendulum_lowfrict.csv',
        'double_pendulum_highfrict.csv'
    ]

    for f in files:
        # Load processed data
        df = pd.read_csv('data/processed/'+f)
        vlim = df[cols].abs().max().to_numpy()

        show_heatmap(df, cols, names = status_names,  vlim=vlim, file_name=f'plots/{f.split(".")[0]}_heatmap')
    
    show_example('data/processed/double_pendulum_lowfrict.csv', id=example_id, animate=True)

if __name__ == '__main__':
    main()
    
