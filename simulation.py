import matplotlib.pyplot as plt

class Simulator: 
    def __init__(self):
        pass

    def solver(self, states, model, dt,  method: str = 'fe'):
        if method == 'rk':
            k1 = model.dynamics(states=states)
            k2= model.dynamics(states=states + dt * k1 / 2)
            k3= model.dynamics(states=states + dt * k2 / 2)
            k4= model.dynamics(states=states + dt * k3)

            new_states = states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == 'fe':
            new_states = states + dt * model.dynamics(states=states)
        else:
            raise ValueError("choose a valid method: 'fe' or 'rk'")
        return new_states
    
    def draw_simulation(self, model, states):
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()

        stop_animation = False  

        def on_key(event):
            nonlocal stop_animation
            if event.key == 'q':
                stop_animation = True
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        steps = states.shape[0]

        for i in range(steps):
            if stop_animation:
                break
            
            ax.clear()
            model.animation(states, i, ax)

            plt.draw()
            plt.pause(0.001)

        plt.close(fig)