import pandas as pd
import numpy as np

FRICT_TYPE = 'nofrict'  # Options: 'none', 'viscous', 'coulomb'

# Import full csv dataset (shape: (B*T, C))
df = pd.read_csv(f"data/processed/double_pendulum_{FRICT_TYPE}.csv")  

# Metric on energy evolution
m1 = df['m1'].iloc[0]
m2 = df['m2'].iloc[0]
l1 = df['l1'].iloc[0]
l2 = df['l2'].iloc[0]
b1 = df['b1'].iloc[0]
b2 = df['b2'].iloc[0]

# x and x_dot columns definition
x_cols = ['q1', 'q2','q1_dot', 'q2_dot']
x_dot_dot_cols = ['q1_dot_dot', 'q2_dot_dot']
x_dot_cols = ['q1_dot', 'q2_dot', 'q1_dot_dot', 'q2_dot_dot']
# Extract x and x_dot (shape: (B*T, 4))
x = []
x_dot = []
x_dot_dot = []
energy = []

for step in df['step'].unique():
    df_step = df[df['step'] == step]
    x_step = df_step[x_cols].values.astype(np.float32)
    x_dot_step = df_step[x_dot_cols].values.astype(np.float32)
    x_dot_dot_step = df_step[x_dot_dot_cols].values.astype(np.float32)
    energy_step = df_step['E'].values.astype(np.float32)
    x.append(x_step)
    x_dot.append(x_dot_step)
    x_dot_dot.append(x_dot_dot_step)
    energy.append(energy_step)

# stack along time dimension 
x = np.stack(x)             # Shape: (T, B, 4)
x_dot = np.stack(x_dot)         # Shape: (T, B, 4)
x_dot_dot = np.stack(x_dot_dot)     # Shape: (T, B, 2)
energy = np.stack(energy)   # Shape: (T, B)


print(x[0:5, 5, :])