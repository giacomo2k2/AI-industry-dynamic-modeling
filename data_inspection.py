import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
This module is responsible for visualizing the state-space evolution of 
mechanical systems (specifically double pendulums) across multiple simulation 
instances. It converts tabular data into heatmap visualizations to easily 
identify patterns, stability, or anomalies in the dataset.
"""

def plot_pivot_subplots(df, inputs: list, pendulum_id: str ='id_pendulum', time: str ='time_step',
                        vmin: float =-5, vmax: float =5, figsize: tuple =(12,6)) -> None:
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
                              Defaults to 'time_step'.
        vmin (float, optional): Minimum value for colormap anchoring (for visual consistency). 
                                Defaults to -5.
        vmax (float, optional): Maximum value for colormap anchoring. 
                                Defaults to 5.
        figsize (tuple, optional): The dimensions of the generated figure (width, height). 
                                   Defaults to (12, 6).

                                   ---
    Returns:
        None: Displays the plot window via plt.show().
    """
    # Initialize a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)

    # Zip flattens the 2D axes array to iterate linearly along with the input variables
    for ax, inp in zip(axes.flatten(), inputs):
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

if __name__ == '__main__':
    # Load processed data
    df = pd.read_csv('data/processed/double_pendulum_nofrict.csv')

    # Define the states
    cols = ['o1', 'o2', 'o1_dot', 'o2_dot']

    # Generate the pivot subplots
    plot_pivot_subplots(df, cols)
