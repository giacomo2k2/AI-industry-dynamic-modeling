import matplotlib.pyplot as plt
import numpy as np

class Simulator: 
    """
    Physics engine responsible for numerical integration and visualization 
    of dynamical systems.

    This class decouples the integration logic from the physical model. 
    It expects a 'model' object that implements 'dynamics' (equations of motion) 
    and 'animation' (visualization logic).
    """

    def __init__(self):
        """Initializes the Simulator instance devoted for the numerical integration and visualization of dynamical system."""
        pass

    def solver(self, states: np.ndarray, model: object, dt: float,  method: str = 'fe') -> np.ndarray:
        """
        Advances the system state by a single time step using numerical integration.

        ---
        Args:
            states (np.ndarray): The current state vector of the system.
            model (object): The physical model instance.
            dt (float): The time step size for integration.
            method (str, optional): The integration scheme to use. 
                                    - Options: 
                                        - 'fe' (Forward Euler)
                                        - 'rk' (Runge-Kutta 4)
                                    - Defaults to 'fe'.
        
        ---
        Returns:
            np.ndarray: The new state vector after time dt.
        """
        if method == 'rk':
            k1 = model.dynamics(states=states)
            k2= model.dynamics(states=states + dt * k1 / 2)
            k3= model.dynamics(states=states + dt * k2 / 2)
            k4= model.dynamics(states=states + dt * k3)

            new_states = states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == 'fe':
            new_states = states + dt * model.dynamics(states=states)
        else:
            raise ValueError("Choose a valid method: 'fe' or 'rk'")
        return new_states
    
    def draw_simulation(self, model: object, states: np.ndarray):
        """
        Draw an animation of the system's trajectory.
        
        ---
        Args:
            model (object): The physical model instance.
            states (np.ndarray): A time-series array of state vectors 
                                 (shape: [num_steps, state_dim]).
        """
        fig, ax = plt.subplots()
        plt.ion()       # Turn on interactive mode
        plt.show()

        # Stop animation flag
        stop_animation = False  

        # Event handler to allow quitting the animation cleanly
        def on_key(event):
            nonlocal stop_animation
            if event.key == 'q':
                stop_animation = True
                plt.close(fig)

        # Connect the key press event to the figure
        fig.canvas.mpl_connect('key_press_event', on_key)

        # States shape
        steps = states.shape[0]

        # Draw animation
        for i in range(steps):
            if stop_animation:
                break
            
            ax.clear()
            model.animation(states, i, ax)

            plt.draw()
            plt.pause(0.001)

        plt.close(fig)