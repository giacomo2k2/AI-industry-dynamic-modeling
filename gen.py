import model as dyn
from simulation import Simulator
import pandas as pd
import numpy as np


def main():
    """
    Main execution function for generating the model dataset.
    
    It simulates multiple pendulum instances with random initial conditions,
    integrates the dynamics using a Simulator, and saves the trajectory data 
    (positions, velocities, and 'next state' values) to a CSV file.
    """

    # Create a dataframe with the columns: [ 'id', 'm1', 'm2', 'l1', 'l2', 'g', 'b1', 'b2', 'time_step', 'o1', 'o2', 'o1_dot', 'o2_dot', 'o1_dot_dot', 'o2_dot_dot']
    df = pd.DataFrame(columns=[
        'id_pendulum', 'm1', 'm2', 'l1', 'l2', 'b1', 'b2', 'g',  
        'time_step', 'o1', 'o2', 'o1_dot', 'o2_dot', 'o1_dot_dot', 'o2_dot_dot'
    ])
    
    n_pendulums = 100       # Number of pendulums in the dataset
    steps = 1000            # Number of steps
    dt = 0.01               # Timestep
    
    # Random seed
    np.random.seed(42)

    # Pendulum creation
    pendulum = dyn.DoublePendulum(
        m1=2.2, m2=1.8, l1=1.2, l2=1, b1=0.3, b2=0.3
    )

    # Pendulums runs generation
    for i in range(n_pendulums):
        print(f"Pendulum run: {i}")

        # Random initial condition
        pendulum.random_init()
        
        # Simulator object
        simulator = Simulator()

        # Create the states time vector
        states = np.zeros((steps+1, 4))
        states[0,:]= pendulum.initial_state
        
        # Create the dataset
        for step in range(0, steps):
            # Integrate the dynamics
            states[step+1,:]=simulator.solver(states[step,:], 
                                              pendulum, 
                                              dt,
                                              method='rk')
            
            # Concatenate the states to the DataFrame
            df = pd.concat([df, pd.DataFrame([{
                    'id_pendulum': i,
                    'm1': pendulum.m1,
                    'm2': pendulum.m2,
                    'l1': pendulum.l1,
                    'l2': pendulum.l2,
                    'b1': pendulum.b1,
                    'b2': pendulum.b2,
                    'g': pendulum.g,
                    'time_step': step * dt,
                    'o1': states[step, 0],
                    'o2': states[step, 2],
                    'o1_dot': states[step, 1],
                    'o2_dot': states[step, 3],
                    'o1_dot_dot': states[step+1, 0],
                    'o2_dot_dot': states[step+1, 1]
            }])], ignore_index=True)
        
    # Save dataframe to csv
    df.to_csv('data/raw/double_pendulum_lowfrict.csv', index=False)


if __name__ == '__main__':
    main()
    

