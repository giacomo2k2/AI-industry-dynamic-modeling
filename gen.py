import numpy as np
import pandas as pd

from model import DoublePendulum
from simulation import Simulator


def main():

    n_pendulums = 100
    steps = 1000
    dt = 0.01

    np.random.seed(42)

    # Single pendulum instance reused; only init changes
    pendulum = DoublePendulum(
        m1=2.2, m2=1.8, l1=1.2, l2=1.0, b1=0.0, b2=0.0
    )

    simulator = Simulator(dt)

    rows = []

    for i in range(n_pendulums):
        print(f"Pendulum run: {i}")

        pendulum.random_init()

        states = np.zeros((steps + 1, 4))
        states[0, :] = pendulum.initial_state

        for step in range(steps):
            q_dot = pendulum.dynamics(states[step, :])
            T, U, E = pendulum.energy(states[step, :])
            states[step+1, :] = simulator.solver(states[step, :], pendulum, method="rk")

            rows.append({
                'id_pendulum': i,
                'm1': pendulum.m1,
                'm2': pendulum.m2,
                'l1': pendulum.l1,
                'l2': pendulum.l2,
                'b1': pendulum.b1,
                'b2': pendulum.b2,
                'g': pendulum.g,
                'time_step': step * dt,
                'q1': states[step, 0],
                'q2': states[step, 2],
                'q1_dot': states[step, 1],
                'q2_dot': states[step, 3],
                'q1_dot_dot': q_dot[1],
                'q2_dot_dot': q_dot[3],
                'T': T,
                'U': U,
                'E': E
            })


    df = pd.DataFrame(rows)
    df.to_csv('data/raw/double_pendulum_nofrict.csv', index=False)

if __name__ == '__main__':
    main()