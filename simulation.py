import matplotlib.pyplot as plt
import numpy as np
import time

plt.rc('font', family='serif')

class Simulator: 
    """
    Physics engine responsible for numerical integration and visualization 
    of dynamical systems.

    This class decouples the integration logic from the physical model. 
    It expects a 'model' object that implements 'dynamics' (equations of motion) 
    and 'animation' (visualization logic).
    """

    def __init__(self, dt: float):
        """Initializes the Simulator instance devoted for the numerical integration and visualization of dynamical system."""
        self.dt = dt

    def solver(self, states: np.ndarray, model: object,  method: str = 'fe') -> np.ndarray:
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
            k2= model.dynamics(states=states + self.dt * k1 / 2)
            k3= model.dynamics(states=states + self.dt * k2 / 2)
            k4= model.dynamics(states=states + self.dt * k3)

            new_states = states + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == 'fe':
            new_states = states + self.dt * model.dynamics(states=states)
        else:
            raise ValueError("Choose a valid method: 'fe' or 'rk'")
        return new_states
    
       
    def draw_simulation(self, model: object, states: np.ndarray, target_fps: int = 30):
        """
        Draw an animation of the system's trajectory.
        
        ---
        Args:
            model (object): The physical model instance.
            states (np.ndarray): A time-series array of state vectors 
                                 (shape: [num_steps, state_dim]).
            target_fps (int): Set a target FPS rate
        """
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()

        # Calculate time per frame
        render_interval = 1.0 / target_fps
        last_render_time = -render_interval  # Ensure we draw the first frame

        start_time = time.time()
        
        for k, state in enumerate(states):
            # The exact time this specific state represents
            current_sim_time = k * self.dt
            
            # --- SYNCHRONIZATION ---
            # Check actual wall-clock time elapsed
            real_time_elapsed = time.time() - start_time
            
            # How long we need to wait to match real-time
            time_to_wait = current_sim_time - real_time_elapsed
            
            if time_to_wait > 0:
                # We are ahead of schedule (computer is fast). Sleep to sync.
                time.sleep(time_to_wait)
            
            # --- RENDERING (Frame Skipping Logic) ---
            # Only draw if enough time has passed since the last draw to match target_fps.
            # AND do not draw if we are significantly lagging behind (time_to_wait < -0.1)
            now = time.time()
            if now - last_render_time >= render_interval:
                
                # If we are lagging significantly (drawing is too slow), 
                # we skip the draw to let the loop catch up to real-time.
                if time_to_wait > -0.05: 
                    ax.clear()
                    model.animation(state, ax)
                    
                    # Add timestamp
                    ax.text(
                        0.02, 0.95,
                        f"t = {current_sim_time:.2f} s",
                        transform=ax.transAxes,
                        fontsize=12,
                        verticalalignment='top'
                    )
                    
                    # Update plot
                    plt.draw()
                    plt.pause(0.001)
                    
                    last_render_time = time.time()