import numpy as np
import matplotlib.pyplot as plt

# Set global plot style
plt.rc('font', family='serif')

class DoublePendulum:
    """
    A simulator for a Double Pendulum system.

    This class implements the equations of motion for a 2-link planar robot 
    (double pendulum) using the standard manipulator equation:

        M(q)q_dd + C(q, q_d)q_d + D q_d + G(q) = u

    State Vector Convention:
        The state vector is ordered as: [theta1, theta1_dot, theta2, theta2_dot]
        - theta1 (q1): Angle of the first link relative to the vertical.
        - theta2 (q2): Relative angle of the second link with respect to the first link.
    """

    def __init__(self,
                 initial_state: np.ndarray = np.array([np.pi/4, 0, np.pi/2, 0]),
                 m1: float = 1, 
                 m2: float = 2, 
                 l1: float = 1.5, 
                 l2: float = 1.5,
                 b1: float = 2, 
                 b2: float = 2,
                 g: float = 9.81):

        """
        Initializes the Double Pendulum parameters.

        Args:
            initial_state (np.ndarray): Initial state vector [q1, q1_dot, q2, q2_dot].
            m1 (float): Mass of the first link (kg).
            m2 (float): Mass of the second link (kg).
            l1 (float): Length of the first link (m).
            l2 (float): Length of the second link (m).
            b1 (float): Damping coefficient for the first joint.
            b2 (float): Damping coefficient for the second joint.
            g (float): Gravitational acceleration (m/s^2).
        """
        
        self.initial_state = initial_state
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.b1 = b1
        self.b2 = b2
        self.g = g

    def random_params(self):
        """
        Randomizes the physical parameters (mass, length, damping) of the system.
        """
        self.m1 = np.random.uniform(1.0, 3.0)
        self.m2 = np.random.uniform(1.0, 3.0)
        self.l1 = np.random.uniform(0.1, 1.0)
        self.l2 = np.random.uniform(0.1, 1.0)
        self.b1 = np.random.uniform(1.0, 5.0)
        self.b2 = np.random.uniform(1.0, 5.0)
    
    def random_init(self):
        """
        Randomizes the initial state of the pendulum.

        ---
        Sets:
         - random angles between [-pi, pi]
         - random velocities between [-0.5, 0.5].

        Updates self.initial_state.
        """
        q1_init = np.random.uniform(-np.pi, np.pi)
        q1_dot_init = np.random.uniform(-2, 2)
        q2_init = np.random.uniform(-np.pi, np.pi)
        q2_dot_init = np.random.uniform(-2, 2)

        self.initial_state = np.array([q1_init, q1_dot_init, q2_init, q2_dot_init])

    def dynamics(self, states: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """
        Computes the time derivative of the state (forward dynamics).

        Solves the Lagrangian equation of motion for acceleration (q_ddot).
        Equation:
         
           M(q) * q_ddot + C(q, q_dot) + D * q_dot + G(q) = u

        ---
        Args:
            states (np.ndarray): The current state vector of shape (4,).
            u (np.ndarray, optional): Control input (torque) vector of shape (2,1).

        ---
        Returns:
            states_dot (np.ndarray): The state derivatives [q1_dot, q1_ddot, q2_dot, q2_ddot] vector of shape (4,).
        """
        
        # Select zero input
        if u is None:
            u = np.zeros((2, 1))

        # Unpack state variables
        q1 = states[0]
        q1_dot = states[1]
        q2 = states[2]
        q2_dot = states[3]

        # Unpack physical parameters for readability
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        b1, b2 = self.b1, self.b2
        
        # === 1. Inertia Matrix M(q) ===
        # Describes how mass is distributed. Symmetric and positive definite
        M = np.array([
            [(m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(q2), m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)],
            [m2 * l2**2 + m2 *l1 *l2 * np.cos(q2), m2 * l2**2]
        ])
        
        # === 2. Coriolis and Centrifugal Forces Vector -> c(q, q_dot) = C(o, q_dot)*q_dot ===
        # Represents fictitious forces arising from rotation. Strongly non linear.
        c = np.array([
            [-m2 *l1*l2*np.sin(q2) * (2*q1_dot + q2_dot) * q2_dot],
            [m2 * l1*l2*np.sin(q2) * q1_dot**2]
        ])

        # === 3. Viscous damping matrix D ===
        # Represents viscous friction in the joints.
        D = np.array([
            [b1, 0],
            [0, b2]
        ])
    
        # === 4. Gravity term vector === 
        # Represents potential energy forces.
        G = np.array([
            [(m1+m2)*g*l1*np.sin(q1) +  m2*g*l2*np.sin(q1 + q2)], 
            [m2*g*l2*np.sin(q1 + q2)]
        ])

        # States derivatives
        q_dot = np.array([
            [q1_dot], 
            [q2_dot]
            ])

        # Model's dynamic
        b = u - c - D @ q_dot - G 
        q_dot_dot = np.linalg.solve(M, b)
        # q_dot_dot = - np.linalg.pinv(M) @ (c + D @ q_dot + G - u)

        

        det = np.linalg.det(M)
        #print("===== Mass matrix inversion =====")
        #print(np.linalg.pinv(M))
        #print("===== Mass matrix determinant ====")
        #print(det)

        assert q_dot_dot.shape == (2, 1)

        q1_dot_dot = float(q_dot_dot[0, 0])
        q2_dot_dot = float(q_dot_dot[1, 0])

        # Pack the derivative of the full state
        states_dot = np.array([q1_dot, q1_dot_dot, q2_dot, q2_dot_dot])
        assert states_dot.shape == (4,)

        return states_dot

    def kinetic_energy(self, states: np.ndarray):
        """
        Computes the Kinetic Energy (T) of the system.
        
        T = 0.5 * q_dot.T * M(q) * q_dot

        ---
        Args:
            q (tuple): Joint angles (q1, q2).
            q_dot (tuple): Joint velocities (q1_dot, q2_dot).

        --- 
        Returns:
            T (float): Total kinetic energy in Joules.
        """
        
        q1 = states[0]
        q1_dot = states[1]  
        q2 = states[2]
        q2_dot = states[3]


        # Term associated with link 1 and coupling
        T = 0.5 * (self.m1 * self.l1**2 + self.m2 * self.l1**2 + self.m2 * self.l2**2 + 2 * self.m2 * self.l1 * self.l2 * np.cos(q2)) * q1_dot**2
        
        # Term associated with link 2 independent motion
        T += 0.5 * self.m2 * self.l2**2 * q2_dot**2

        # Term associated with coupling velocities
        T += 0.5 * (2 * self.m2 * self.l2**2 + 2 * self.m2 * self.l1 * self.l2 * np.cos(q2)) * q1_dot * q2_dot
        
        return T
    
    def potential_energy(self, states: np.ndarray):
        """
        Computes the Potential Energy (V) of the system.

        ---
        Args:
            q (tuple): Joint angles (q1, q2).
            q_dot (tuple): Joint velocities (unused here, kept for API consistency).

        ---
        Returns:
            V (float): Total potential energy in Joules.
        """
        q1 = states[0]
        q2 = states[2]
        
        
        V = self.m1 * self.g * self.l1 * (1-np.cos(q1)) + self.m2 * self.g * (self.l1 * (1-np.cos(q1)) + self.l2*(1- np.cos(q1 + q2))) 

        #V = -m1 * g * l1 * np.cos(q1) - m2 * g * ( l1 * np.cos(q1) +l2* np.cos(q1 + q2)) 
        return V

    def energy(self, states: np.ndarray):
        """        
        Lagrangian L = T - V

        Hamiltonian (Total Energy) H = T + V (conservative systems)

        ---
        Args:
            q (tuple): (q1, q2)
            q_dot (tuple): (q1_dot, q2_dot)

        ---
        Returns:
            tuple: (Kinetic Energy T, Potential Energy V, Total Energy E)
        """
        T = self.kinetic_energy(states)
        V = self.potential_energy(states)

        return T, V, T + V

    def animation(self, states: np.ndarray, ax: plt.Axes):
        """
        Helper function to render a single frame of the double pendulum.

        ---
        Args:
            states (np.ndarray): The full history of states.
            ax (matplotlib.axes.Axes): The plot axis to draw on.
        """
        # Extract joints position
        q1 = states[0]
        q2 = states[2]
        
        # Compute cartesians position (Forward Kinematics)
        p1 = np.array([self.l1 * np.sin(q1), -self.l1 * np.cos(q1)])
        p2 = np.array([self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2), -self.l1 * np.cos(q1) - self.l2 * np.cos(q1 + q2)])

        # Limit for total pendolum length
        lim_size = (self.l1 + self.l2) * 1.5

        # Plot settings
        ax.set_xlim([-lim_size, lim_size])
        ax.set_ylim([-lim_size, lim_size])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Double Pendulum')

        # === Draw Links and Joints ===
        # Link 1
        ax.plot([0.0, p1[0]], [0.0, p1[1]], 'grey', linewidth=2.5, label='Link 1')
        ax.scatter([0.0], [0.0], color='black', s=40, zorder=5) # Base pivot

        # Link 2
        ax.scatter(p1[0], p1[1], color='blue', s=40, zorder=5)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', linewidth=2.5, label='Link 2')
        ax.scatter(p2[0], p2[1], color='red', s=40, zorder=5)

        # Legend
        ax.legend(loc='upper right')
   
    def energy_plot(self, states: np.ndarray, dt:float):
        """
        Plots the energy evolution over the simulation trajectory.

        ---
        Args:
            states (np.ndarray): Array of shape (N, 4) containing the trajectory.
            dt (float): Time step duration.
        """
        kinetics = []
        potentials = []
        total_energy = []

        n_states = states.shape[0]

        for i in range(n_states):
            q = (states[i,0], states[i,2])
            q_dot = (states[i,1], states[i,3])

            T, V, E = self.energy(q, q_dot)
         
            kinetics.append(T)
            potentials.append(V)
            total_energy.append(E)

        time = np.arange(0, n_states * dt, dt)

        plt.figure(figsize=(10, 6))
        plt.plot(time, kinetics, label='Kinetic Energy')
        plt.plot(time, potentials, label='Potential Energy')
        plt.plot(time, total_energy, label='Total Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Energies')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()  
            