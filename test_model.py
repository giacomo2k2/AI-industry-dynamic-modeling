import numpy as np
import matplotlib.pyplot as plt
from models import DoublePendulum
from simulation import Simulator
from models import kinetic_energy, potential_energy
from preprocessing import energy_plot

# Set global plot style
plt.rc('font', family='serif')

def main():

    doublePendulum = DoublePendulum()
    simulator = Simulator()

    doublePendulum.random_init()


    # doublePendulum.m1 = 1.7381329868016133
    # doublePendulum.m2 = 1.0232574489051587
    # doublePendulum.l1 = 0.24781431839479579
    # doublePendulum.l2 = 0.17996002096378658
    # doublePendulum.b1 = 3.9022998222134895
    # doublePendulum.b2 = 4.021174216448582
    # doublePendulum.initial_state = np.array([ 0.57447934, -0.29190807, -2.17704057,  0.38309615])

    #doublePendulum.b1 = 0.0
    #doublePendulum.b2 = 0.0
    steps = 1000
    states = np.zeros((steps, 4))
    
    states[0,:] = doublePendulum.initial_state
    dt = 0.01


    dets = []

    for i in range(steps-1):
        states_dot = doublePendulum.dynamics(states=states[i,:])
        states[i+1,:] = simulator.solver(states[i,:], states_dot, dt, doublePendulum, method='rk')
        
    
    print(f"Masses: m1={doublePendulum.m1}, m2={doublePendulum.m2}")
    print(f"Lengths: l1={doublePendulum.l1}, l2={doublePendulum.l2}")
    print(f"Friction: b1={doublePendulum.b1}, b2={doublePendulum.b2}")
    print(f"Initial state: {doublePendulum.initial_state}")
    energy_plot(states, 
                (doublePendulum.m1, doublePendulum.m2),
                (doublePendulum.l1, doublePendulum.l2),
                dt)
        
   
    
    simulator.draw_simulation(doublePendulum, states)



if __name__ == '__main__': 
    main()