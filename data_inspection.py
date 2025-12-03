import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from simulation import Simulator
from model import DoublePendulum

"""
This module is responsible for visualizing the state-space evolution of 
mechanical systems (specifically double pendulums) across multiple simulation 
instances. It converts tabular data into heatmap visualizations to easily 
identify patterns, stability, or anomalies in the dataset.
"""

# ========================================================
# ================== SHOW THE DATASET ==================
# ========================================================


def show_status_stats(df: pd.DataFrame, filename  : str = '', save: bool = False):

    """
    Compute variance of each state across all pendulums for each time step
    """
    
    time_steps = df['time_step'].unique()
    stds = np.zeros((time_steps.shape[0], 4))
    means = np.zeros((time_steps.shape[0], 4))
    
    for i, t in enumerate(time_steps):
        step_data = df[df['time_step'] == t]
        states = step_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()
        stds[i, :] = np.std(states, axis=0)
        means[i, :] = np.mean(states, axis=0)
       
    names = ['o1', 'o1_dot', 'o2', 'o2_dot']
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for (i, a) in enumerate(ax.flatten()):
        a.plot(stds[:, i])
        a.plot(means[:, i], color='orange')
        a.grid()
        a.legend(['Std', 'Mean'])
        a.set_title(f'{names[i]}')

    plt.tight_layout()
    plt.show()

    if save:
        # save the plot as png
        fig.savefig('plots/variance_means_' + filename +'.png')

def show_energy_stats(df:pd.DataFrame, threshold:float=None, filename:str='', save:bool = False):
    """
    Keep onlty the time steps lower than a threshold
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
    plt.legend(['std', 'mean'])
    plt.tight_layout()
    plt.show()

    if save:
    #save the plot as png
        fig.savefig('plots/energy_stats_' + filename + '.png')
    

def show_bins(df: pd.DataFrame, id:int = -1, save: bool = False, filename: str = ''):

    # Plot the histogram for o1, o2, o1_dot, o2_dot

    if (id != -1):
        #select a pendulum
        pendulum_id = id
        pendulum_data = df[df['id_pendulum'] == pendulum_id]
    else:
        pendulum_data = df

    # extract states
    states = pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()


    # bin1 : bin of angles for o1
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    configs = [
        (states[:, 0], (-np.pi, np.pi),    "o1",     "o1 [rad]"),
        (states[:, 1], (-15, 15),          "o1_dot", "o1_dot [rad/s]"),
        (states[:, 2], (-np.pi, np.pi),    "o2",     "o2 [rad]"),
        (states[:, 3], (-15, 15),          "o2_dot", "o2_dot [rad/s]")
    ]

    for ax, (data, rng, title, xlabel) in zip(axes.ravel(), configs):
        ax.hist(data, bins=100, range=rng, density=True)
        ax.set_title(f"Histogram of {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")

    fig.tight_layout()
    plt.show()


    if save and id == -1:
        # save the plot as png
        fig.savefig('plots/histogram_pendulum_' + filename + '.png')



def show_heatmap(df, inputs: list, pendulum_id: str ='id_pendulum', time: str ='time_step',
                        vmin_q: float =-np.pi, vmax_q: float =np.pi, vmin_q_dot: float =-10, 
                        vmax_q_dot: float =10, figsize: tuple =(12,6), file_name:str = None) -> None:
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
        vmin_q (float, optional): Minimum value for colormap anchoring w.r.t. angular position
        vmax_q (float, optional): Maximum value for colormap anchoring w.r.t. angular position
        vmin_q_dot (float, optional): Minimum value for colormap anchoring w.r.t. angular velocity
        vmax_q_dot (float, optional): Maximum value for colormap anchoring w.r.t. angular velocity
        figsize (tuple, optional): The dimensions of the generated figure (width, height). 
        file_name (str, optional): Name necessary to save the subplot image.

    ---
    Returns:
        None: Displays the plot window via plt.show().
    """
    # Initialize a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)

    # Zip flattens the 2D axes array to iterate linearly along with the input variables
    for ax, inp in zip(axes.flatten(), inputs):
        # Determine vmin and vmax based on variable type
        if 'dot' in inp: #TODO make this if better
            vmin, vmax = vmin_q_dot, vmax_q_dot
        else:
            vmin, vmax = vmin_q, vmax_q
            
        # Pivot table: Rows = Simulation ID, Cols = Time, Values = State Variable
        pivot = df.pivot(index=pendulum_id, columns=time, values=inp)

        # Plot heatmap
        # aspect='auto' ensures the pixels fill the plot area regardless of data shape
        im = ax.imshow(pivot, aspect='auto', cmap='RdBu', vmin=vmin, vmax=vmax)

        # Decoration
        ax.set_title(inp)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pendulum ID")
        
    plt.tight_layout()
    plt.show()
    
    # Save the image as a png file
    if not file_name:
        file_name = 'subplots'+'_'.join(inputs)
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

        simulator.draw_simulation(pendulum, pendulum_data[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy())

def main():
    example_id = 55
    # Define the states
    cols = ['o1', 'o2', 'o1_dot', 'o2_dot']

    files = [
        'double_pendulum_nofrict.csv',
        'double_pendulum_lowfrict.csv',
        'double_pendulum_highfrict.csv'
    ]

    for f in files:
        # Load processed data
        df = pd.read_csv('data/processed/'+f)

        show_heatmap(df, cols, file_name=f'plots/{f.split(".")[0]}_heatmap')
    
    show_example('data/raw/double_pendulum_nofrict.csv', id=example_id, animate=True)

if __name__ == '__main__':
    main()
    
