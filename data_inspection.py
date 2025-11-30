import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_pivot_subplots(df, inputs, pendulum_id='id_pendulum', time='time_step',
                        vmin=-5, vmax=5, figsize=(12,6)):

    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)

    for ax, inp in zip(axes.flatten(), inputs):
        pivot = df.pivot(index=pendulum_id, columns=time, values=inp)

        im = ax.imshow(pivot, aspect='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(inp)
        ax.set_xlabel("time")
        ax.set_ylabel("pendulum id")
        
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    df = pd.read_csv('data/processed/double_pendulum_fixed.csv')
    cols = ['o1', 'o2', 'o1_dot', 'o2_dot']
    plot_pivot_subplots(df, cols)
