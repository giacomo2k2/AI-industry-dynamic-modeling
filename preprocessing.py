import pandas as pd 
import numpy as np
import dynamics as dyn
from simulation import Simulator
from physics import kinetic_energy, potential_energy
import matplotlib.pyplot as plt

# Show the energy plot of kinetic, potential and total energy given a list of states
def energy_plot(states, 
                masses,
                lengths, 
                dt):
    
    kinetics = []
    potentials = []
    lagrangians = []
    
    m1, m2 = masses
    l1, l2 = lengths

    n_states = states.shape[0]

    for i in range(n_states):

        o = (states[i,0], states[i,1])
        o_dot = (states[i,2], states[i,3])

        T = kinetic_energy(o, o_dot, m1, m2, l1, l2)
        U = potential_energy(o, o_dot, m1, m2, l1, l2)
    

        kinetics.append(T)
        potentials.append(U)
        lagrangians.append(T + U + 20)

    time = np.arange(0, n_states*dt, dt)
    plt.plot(time, kinetics, label='Kinetic Energy')
    plt.plot(time, potentials, label='Potential Energy')
    plt.plot(time, lagrangians, label='Lagrangian Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energies')
    plt.legend()
    plt.show()


def main():

    df = pd.read_csv('double_pendulum_data.csv')

    #select a pendulum
    pendulum_id = 0
    pendulum_data = df[df['id_pendulum'] == pendulum_id]

    #create a numpy array (steps, 4) with the o1, o1_dot, o2, o2_dot
    states = pendulum_data[['o1', 'o2', 'o1_dot', 'o2_dot']].to_numpy()

    print(states[0,:])

    #extract dt
    dt = pendulum_data['time_step'].iloc[1] - pendulum_data['time_step'].iloc[0]

    #extract masses
    m1 = pendulum_data['m1'].iloc[0]
    m2 = pendulum_data['m2'].iloc[0]
    masses = (m1, m2)

    #extract lengths
    l1 = pendulum_data['l1'].iloc[0]
    l2 = pendulum_data['l2'].iloc[0]
    lengths = (l1, l2)

    #plot the energies
    energy_plot(states, masses, lengths, dt)

if __name__ == '__main__':
    main()