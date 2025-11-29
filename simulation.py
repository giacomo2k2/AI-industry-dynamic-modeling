import matplotlib.pyplot as plt

class Simulator: 
    def __init__(self):
        pass

    def solver(self, states, states_dot, dt, method: str = 'fe'):
        if method == 'rk':
            k1 = states_dot
            k2 = self.dynamics(states=states + dt * k1 / 2)
            k3 = self.dynamics(states=states + dt * k2 / 2)
            k4 = self.dynamics(states=states + dt * k3)

            new_states = states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == 'fe':
            new_states = states + dt * states_dot
        else:
            raise ValueError("choose a valid method: 'fe' or 'rk'")
        return new_states
    
    def draw_simulation(self, model, states):
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()

        stop_animation = False  

        def on_key(event):
            if event.key == 'q':
                stop_animation = True
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        steps = states.shape[0]

        for i in range(steps):
            if stop_animation:
                break
            
            ax.clear()
            model.animation(states, i)

            plt.draw()
            plt.pause(0.001)