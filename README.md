# 0. Background

## Double Pendulum Modeling

The double pendulum consists of two masses $m_1$ and $m_2$ connected by rigid, massless rods of lengths $l_1$ and $l_2$. The first mass $m_1$ is attached to a fixed pivot, while $m_2$ is connected to $m_1$ via a second rod. Gravity acts downward with acceleration $g$.

- **Generalized coordinates:**  
  - $\theta_1$: Angle of the first rod from the vertical  
  - $\theta_2$: Angle of the second rod relative to the first

- **Positions:**  
  - $p_{1} = [l_1 \sin(\theta_1),\ l_1 \cos(\theta_1)]$  
  - $p_{2} = [l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2),\ l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2)]$

- **Velocities:**  
  - $v_{1} = [l_1 \cos(\theta_1)\, \dot{\theta}_1,\ -l_1 \sin(\theta_1)\, \dot{\theta}_1]$  
  - $v_{2} = [l_1 \cos(\theta_1)\, \dot{\theta}_1 + l_2 \cos(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2),\ -l_1 \sin(\theta_1)\, \dot{\theta}_1 - l_2 \sin(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)]$

  ## Background 

  - Measuring parameters of a complex physical system can be higly costly due to the use higl level strumentation
  - However, for production these in unsustainable since every product cannot have internal sensors so expensive
  - For this reason, having a model that is able to determine this dynamical parameters starting from the state can save a lot of money
  
   

