import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import seaborn as sns
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

def show_status_stats(df: pd.DataFrame, cols: list, names: list, filename: str = '', save: bool = False):
    """
    Compute variance of each state across all pendulums for each time step.
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
    
    if save and filename:
        fig.savefig(filename)

    plt.show()

def show_energy_stats(df: pd.DataFrame, filename: str = '', save: bool = False):
    """
    Compute for each time step the medium total energy.
    """
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
    
    if save and filename:
        fig.savefig(filename)
        
    plt.show()

def show_bins(
        df: pd.DataFrame,  
        cols: list, names: list, 
        vlim: np.ndarray, 
        save_path: str = None, 
    ):

    pendulum_data = df

    # extract states
    states = pendulum_data[cols].to_numpy()


    # use module on angles
    states[:, 0] = np.mod(states[:, 0] + np.pi, 2 * np.pi) - np.pi
    states[:, 3] = np.mod(states[:, 3] + np.pi, 2 * np.pi) - np.pi


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
        ax.hist(data, bins=200, range=rng, density=True)
        ax.set_title(f"Histogram of {title}")
        ax.set_ylabel("Density")

    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path + '_bins.png')

    plt.show()

def show_heatmap(
        df, 
        cols: list, 
        names: list, 
        vlim: np.ndarray, 
        pendulum_id: str = 'id_pendulum', 
        time: str = 'time_step',
        figsize: tuple = (12, 6), 
        file_name: str = None
    ) -> None:
    """
    Generates a 2x2 grid of heatmaps visualizing the evolution of state variables over time.
    """
    # Initialize a 2x3 subplot grid (matches the 6 variables in cols)
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    fig.suptitle('State Variables Heatmaps Over Time', fontsize=16)
    
    # Zip flattens the 2D axes array to iterate linearly along with the input variables
    for ax, col, name, abs_vlim in zip(axes.flatten(), cols, names, vlim):
        # Pivot table: Rows = Simulation ID, Cols = Time, Values = State Variable
        pivot = df.pivot(index=pendulum_id, columns=time, values=col)

        # Plot heatmap
        im = ax.imshow(pivot, aspect='auto', cmap='RdBu', vmin=-abs_vlim, vmax=abs_vlim)

        # Decoration
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pendulum ID")
        
    plt.tight_layout()
    
    if file_name:
        fig.savefig(file_name + ".png")

    plt.show()

def show_example(path: str, id: int = 0, animate: bool = False):
    """ 
    Plot the energies and draw the animation for a given pendulum from the csv file.
    """
    df = pd.read_csv(path)

    pendulum_data = df[df['id_pendulum'] == id]

    time_arr = pendulum_data['time_step'].to_numpy()
    T = pendulum_data['T'].to_numpy()
    U = pendulum_data['U'].to_numpy()
    E = pendulum_data['E'].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, T, label='Kinetic Energy')
    plt.plot(time_arr, U, label='Potential Energy')
    plt.plot(time_arr, E, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    if animate:
        dt = time_arr[1] - time_arr[0]
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

def show_conditional_distribution(
        df: pd.DataFrame,
        target_col: str,
        condition_col: str,
        condition_bins: int = 10,
        row_col_plots: tuple = (5, 2)
):
    """
    Plots the conditional distribution of a target variable given binned conditions of another variable.
    """
    rows, cols = row_col_plots

    if rows * cols < condition_bins:
        raise ValueError("row_col_plots deve poter accomodare tutti i condition bins.")

    condition_values = df[condition_col]
    target_values = df[target_col]

    # Creazione dei bin
    bins = np.linspace(condition_values.min(), condition_values.max(), condition_bins + 1)
    bin_indices = np.digitize(condition_values, bins) - 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10)) 
    axes = axes.flatten()  

    for i in range(condition_bins):
        ax = axes[i] 

        bin_target_values = target_values[bin_indices == i]
        
        if len(bin_target_values) > 0:
            ax.hist(bin_target_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            ax.set_title(f'Bin {i+1}: {bins[i]:.2f} - {bins[i+1]:.2f}')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Hide subplots that are not used
    for j in range(condition_bins, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Conditional Distribution of {target_col} given {condition_col}', fontsize=16)
    plt.tight_layout() 
    plt.show()


def show_correlation_matrix(
        df: pd.DataFrame, 
        target: str,
        features_cols: list,
        save: bool = False,
        filename: str = ''
    ):
    """
    Plots the correlation matrix for the specified columns in the DataFrame.
    """
    cols = [target] + features_cols
    corr = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    
    if save and filename:
        plt.savefig(filename)
        
    plt.show()
    

def main(
        processed_folder: str,
        example_id: int = 17
):
    # Create a unique timestamped folder for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_plot_dir = os.path.join('plots', f"run_{timestamp}_{processed_folder.split('/')[-1]}")
    os.makedirs(unique_plot_dir, exist_ok=True)
    
    print(f"--- Visualization Started ---")
    print(f"Reading data from: {processed_folder}")
    print(f"Saving all plots to: {unique_plot_dir}")

    # Define the states
    cols = ['q1', 'q1_dot','q1_dot_dot', 'q2', 'q2_dot', 'q2_dot_dot']
    status_names = [r'$q_1$', r'$\dot{q}_1$', r'$\ddot{q}_1$', r'$q_2$', r'$\dot{q}_2$',  r'$\ddot{q}_2$']


    for f in os.listdir(processed_folder):
        file_path = os.path.join(processed_folder, f)
        
        if not os.path.isfile(file_path) or not f.endswith('.csv'):
            continue


        # Load processed data

        df = pd.read_csv(file_path)
        base_name = f.split('.csv')[0]

        # show_conditional_distribution(
        #     df = df, 
        #     target_col='q1_dot_dot', 
        #     condition_col='q1', 
        #     condition_bins=3,
        #     row_col_plots=(1, 3)
        # )

        show_correlation_matrix(
            df = df,
            target = 'q1_dot_dot',
            features_cols = ['q1', 'q1_dot', 'q2', 'q2_dot'],
            save = True,
            filename = os.path.join(unique_plot_dir, f"{base_name}_correlation.png")
        )

        # Compute lim for heatmap

        vlim = df[cols].abs().max().to_numpy()

        # Show heatmap (a heatmap for each state)
        show_heatmap(
            df = df, 
            cols = cols,
            names = status_names,  
            vlim = vlim, 
            file_name = os.path.join(unique_plot_dir, f"{base_name}_heatmap")
        )

        show_bins(
            df = df,
            cols = cols,
            names = status_names,
            vlim = vlim,
            save_path = os.path.join(unique_plot_dir, base_name)
        )

        show_energy_stats(
            df = df,
            filename = os.path.join(unique_plot_dir, f"{base_name}_energy_stats.png"),
            save = True
        )

        show_status_stats(
            df = df,
            cols = cols,
            names = status_names,
            filename = os.path.join(unique_plot_dir, f"{base_name}_variance_means.png"),
            save = True
        )
    

    # Example visualization (not saved to folder by default logic, but displayed)
    example_path = 'data/processed/double_pendulum_nofrict.csv'
    if os.path.exists(example_path):
        show_example(example_path, id=example_id, animate=True)
    else:
        print(f"Example file not found at {example_path}, skipping animation.")


if __name__ == '__main__':

    processed_folder = 'branch1/data_enhanced/processed_EC_WN'
    example_id = 17

    main(
        processed_folder=processed_folder,
        example_id=example_id
    )