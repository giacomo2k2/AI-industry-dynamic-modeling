import numpy as np
import matplotlib.pyplot as plt
from dynamics import DoublePendulum
from simulation import Simulator


def main():

    doublePendulum = DoublePendulum()
    simulator = Simulator()

    initial_state = np.array([np.pi/2, 0.0, np.pi/4, 0.0])
    
    steps = 1000
    states = np.zeros((steps, 4))
 
    states[0,:] = initial_state
    dt = 0.01

    for i in range(steps-1):
    
        states[i+1,:] = simulator.solver(states[i,:], doublePendulum.dynamics(states=states[i,:]), dt)
    
    simulator.draw_simulation(doublePendulum, states)
        

if __name__ == '__main__': 
    main()