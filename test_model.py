import numpy as np
import matplotlib.pyplot as plt
from dynamics import DoublePendulum
from simulation import Simulator
from physics import kinetic_energy, potential_energy
from preprocessing import energy_plot


def main():

    doublePendulum = DoublePendulum()
    simulator = Simulator()

    doublePendulum.random_init()
    
    steps = 1000
    states = np.zeros((steps, 4))
 
    states[0,:] = initial_state
    dt = 0.01

    for i in range(steps-1):
    
        states[i+1,:] = simulator.solver(states[i,:], doublePendulum.dynamics(states=states[i,:]), dt)
    
    simulator.draw_simulation(doublePendulum, states)
    energy_plot(states, 
                (doublePendulum.m1, doublePendulum.m2),
                (doublePendulum.l1, doublePendulum.l2),
                dt)
        

if __name__ == '__main__': 
    main()