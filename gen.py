import dynamics as dyn
from simulation import Simulator
import pandas as pd
import numpy as np


def main():

    # create a dataframe with the columns: [ 'id', 'm1', 'm2', 'l1', 'l2', 'g', 'b1', 'b2', 'time_step', 'o1', 'o2', 'o1_dot', 'o2_dot', 'o1_dot_dot', 'o2_dot_dot']
    df = pd.DataFrame(columns=[
        'id_pendulum', 'm1', 'm2', 'l1', 'l2', 'b1', 'b2', 'g', 
        'time_step', 'o1', 'o2', 'o1_dot', 'o2_dot', 'o1_dot_dot', 'o2_dot_dot'
    ])

    n_pendulums = 50
    steps = 1000
    dt = 0.01
    

    # random seed
    np.random.seed(42)
    pendulum = dyn.DoublePendulum()
    pendulum.random_params()
    
    for i in range(n_pendulums):

        print("Pendulum ", i)

        pendulum.random_init()
        
        simulator = Simulator()

        states = np.zeros((steps+1, 4))
        states[0,:]= pendulum.initial_state
        
        

        for step in range(0, steps):
            #print(f"\t step {step+1}")
            states[step+1,:]=simulator.solver(states[step,:], 
                                              pendulum, 
                                              dt,
                                              method='rk')
            
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
        
    # save dataframe to csv
    df.to_csv('data/double_pendulum_fixed.csv', index=False)


if __name__ == '__main__':

    main()
    

