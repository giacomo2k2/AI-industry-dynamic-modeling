# Data Inspection

## Heatmaps

### Heatmap No Friction

* periodic behaviour, but different periods and also cahotic
* velocities are always in the same range (-> no friction)
* $\dot{o_2}$ presents the highest values, while $o_1$ the lowest ones
* in general, all the samples seem to be significative and no strange patterns are observed

### Heatmap Low Friction
* we notice that the variance across pendulums over time decreases
* the last part of the dataset is not significative 


### Heatmap High Friction
* after 2000 step (2 seconds), the variance is very low because the pendulum converge to equilibrium faster
* after 2 seconds the dataset samples are not really significative



### BRANCH2

# activation
* tanh is not good compared to ReLU famiy
* in ReLU family the best is SiLU which is smoother that ReLU
* In ReLU family ReLU is worst

# BaselineFNN
* Seems good enough for fitting double pendulum. 

# ClassicalODE
* Very good for 1 step ahead learning but not so nice with state and energy evolution as predicted

# NeuralODE
* seems powerful but very slow in training. Instead of learning the next steap ahead 
it learns the entire evolution as predicted. State evolution is good in term of loss but not so accurate in details but energy evolution predicted seems worse


# Dataset
* nofrict and lowfrict seems OK in term of size while highfrict is too small and it causes overfitting
* lowfrict seems to help in state and energy evolution wrt nofrict


## 
