# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import numpy as np 

def kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.81):
  (o1, o2), (o1_dot, o2_dot) = q, q_dot

  T = 0.5 * (m1 * l1**2 + m2 * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(o2)) * o1_dot**2
  
  T += 0.5 * m2 * l2**2 * o2_dot**2

  T += 0.5 * (2 * m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(o2)) * o1_dot * o2_dot
  
  return T

def potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.81):
  (o1, o2), (o1_dot, o2_dot) = q, q_dot

  V = m1 * g * l1 * np.sin(o1) + m2 * g * l2 * np.sin(o1 + o2)
  return V

# Double pendulum lagrangian
def lagrangian_fn(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.81):

  T = kinetic_energy(q, q_dot, m1, m2, l1, l2, g)
  V = potential_energy(q, q_dot, m1, m2, l1, l2, g)
  
  return T - V

# Double pendulum lagrangian
def hamiltonian_fn(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.81):
  (t1, t2), (w1, w2) = q, q_dot

  T = kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.81)
  V = potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.81)
  return T + V
  
# Double pendulum dynamics via analytical forces taken from Diego's blog
def analytical_fn(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
  t1, t2, w1, w2 = state
  a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
  a2 = (l1 / l2) * np.cos(t1 - t2)
  f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
  f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
  g1 = (f1 - a1 * f2) / (1 - a1 * a2)
  g2 = (f2 - a2 * f1) / (1 - a1 * a2)
  return np.stack([w1, w2, g1, g2])
    