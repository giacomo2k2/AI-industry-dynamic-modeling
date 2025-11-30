import numpy as np
import matplotlib.pyplot as plt


class DoublePendulum:

    def __init__(self,
                 initial_state: np.ndarray = np.array([np.pi/4, 0, np.pi/2, 0]),
                 m1: float = 1, 
                 m2: float = 2, 
                 l1: float = 1.5, 
                 l2: float = 1.5,
                 b1: float = 2, 
                 b2: float = 2,
                 g: float = 9.81):
                 
        self.initial_state = initial_state
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.b1 = b1
        self.b2 = b2
        self.g = g

    def random_params(self):

        self.m1 = np.random.uniform(1.0, 3.0)
        self.m2 = np.random.uniform(1.0, 3.0)
        self.l1 = np.random.uniform(0.1, 1.0)
        self.l2 = np.random.uniform(0.1, 1.0)
        self.b1 = np.random.uniform(1.0, 5.0)
        self.b2 = np.random.uniform(1.0, 5.0)

               
    def random_init(self):
        o1_init = np.random.uniform(-np.pi, np.pi)
        o1_dot_init = np.random.uniform(-0.5, 0.5)
        o2_init = np.random.uniform(-np.pi, np.pi)
        o2_dot_init = np.random.uniform(-0.5, 0.5)

        self.initial_state = np.array([o1_init, o1_dot_init, o2_init, o2_dot_init])

    def dynamics(self, states: np.ndarray, u: np.ndarray = None):
        
        """Compute state derivatives given current state.

        states: array-like of shape (4,) -> [o1, o1_dot, o2, o2_dot]
        inputs: optional array-like shape (2,1)
        returns: np.ndarray shape (4,) -> [o1_dot, o1_ddot, o2_dot, o2_ddot]
        """

        if u is None:
            u = np.zeros((2, 1))

        o1 = states[0]
        o1_dot = states[1]
        o2 = states[2]
        o2_dot = states[3]

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g
        b1 = self.b1
        b2 = self.b2
        
        # Inertia matrix
        M = np.array([
            [(m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(o2), m2 * l2**2 + m2 * l1 * l2 * np.cos(o2)],
            [m2 * l2**2 + m2 *l1 *l2 * np.cos(o2), m2 * l2**2]
        ])
        
        # Centrifugal and coriolis effect Vector -> c(o, o_dot) = C(o, o_dot)*o_dot
        c = np.array([
            [-m2 *l1*l2*np.sin(o2) * (2*o1_dot + o2_dot) * o2_dot],
            [m2 * l1*l2*np.sin(o2) * o1_dot**2]
        ])

        # Viscous damping matrix
        D = np.array([
            [b1, 0],
            [0, b2]
        ])
    
        # Gravity term vector
        G = np.array([
            [(m1+m2)*g*l1*np.sin(o1) +  m2*g*l2*np.sin(o1 + o2)], 
            [m2*g*l2*np.sin(o1 + o2)]
        ])

        # States derivatives
        o_dot = np.array([
            [o1_dot], 
            [o2_dot]
            ])

        # Model's dynamic
        b = u - c - D @ o_dot - G 
        o_dot_dot = np.linalg.solve(M, b)


        #o_dot_dot = - np.linalg.pinv(M) @ (c + D @ o_dot + G - u)

        det = np.linalg.det(M)
        #print("===== Mass matrix inversion =====")
        #print(np.linalg.pinv(M))
        #print("===== Mass matrix determinant ====")
        #print(det)
        assert o_dot_dot.shape == (2, 1)

        o1_dot_dot = float(o_dot_dot[0, 0])
        o2_dot_dot = float(o_dot_dot[1, 0])

        states_dot = np.array([o1_dot, o1_dot_dot, o2_dot, o2_dot_dot])
        assert states_dot.shape == (4,)

        return states_dot

    def kinetic_energy(self, q, q_dot):
        (o1, o2), (o1_dot, o2_dot) = q, q_dot

        T = 0.5 * (self.m1 * self.l1**2 + self.m2 * self.l1**2 + self.m2 * self.l2**2 + 2 * self.m2 * self.l1 * self.l2 * np.cos(o2)) * o1_dot**2
        
        T += 0.5 * self.m2 * self.l2**2 * o2_dot**2

        T += 0.5 * (2 * self.m2 * self.l2**2 + 2 * self.m2 * self.l1 * self.l2 * np.cos(o2)) * o1_dot * o2_dot
        
        return T
    
    def potential_energy(self, q, q_dot):
        (o1, o2), (o1_dot, o2_dot) = q, q_dot
        
        V = self.m1 * self.g * self.l1 * (1-np.cos(o1)) + self.m2 * self.g * (self.l1 * (1-np.cos(o1)) + self.l2*(1- np.cos(o1 + o2))) 
        #V = -m1 * g * l1 * np.cos(o1) - m2 * g * ( l1 * np.cos(o1) +l2* np.cos(o1 + o2)) 
        return V


    # Double pendulum lagrangian
    def energy(self, q, q_dot):
        T = self.kinetic_energy(q, q_dot)
        V = self.potential_energy(q, q_dot)

        return T, V, T + V



    def animation(self, states, iter, ax):
        # Extract joints position
        o1 = states[iter, 0]
        o2 = states[iter, 2]
        
        # Compute cartesians position
        p1 = np.array([self.l1 * np.sin(o1), -self.l1 * np.cos(o1)])
        p2 = np.array([self.l1 * np.sin(o1) + self.l2 * np.sin(o1 + o2), -self.l1 * np.cos(o1) - self.l2 * np.cos(o1 + o2)])

        # Limit for total pendolum length
        lim_size = (self.l1 + self.l2) * 1.5

        # Plot settings
        ax.set_xlim([-lim_size, lim_size])
        ax.set_ylim([-lim_size, lim_size])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Double Pendulum - Step {iter}')

        # Plot pendolum motion
        ax.plot([0.0, p1[0]], [0.0, p1[1]], 'grey', linewidth=2.5, label='Link 1')
        ax.scatter([0.0], [0.0], color='black', s=40, zorder=5)
        ax.scatter(p1[0], p1[1], color='blue', s=40, zorder=5)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', linewidth=2.5, label='Link 2')
        ax.scatter(p2[0], p2[1], color='red', s=40, zorder=5)

        # Legend
        ax.legend(loc='upper right')


   
    def energy_plot(self, states, dt):
        
        kinetics = []
        potentials = []
        total_energy = []

        n_states = states.shape[0]

        for i in range(n_states):

            o = (states[i,0], states[i,2])
            o_dot = (states[i,1], states[i,3])

            T, V, E = self.energy(o, o_dot)
         
            kinetics.append(T)
            potentials.append(V)
            total_energy.append(E)

        time = np.arange(0, n_states*dt, dt)
        plt.plot(time, kinetics, label='Kinetic Energy')
        plt.plot(time, potentials, label='Potential Energy')
        plt.plot(time, total_energy, label='Total Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Energies')
        plt.legend()
        plt.show()  
            