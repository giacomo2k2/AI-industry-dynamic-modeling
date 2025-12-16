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


# TODO
* Undestand why LagrangianNN doesn't accurately score the energy
* Model single pendulum to achieve more consideration in case of non-cahotic behaviour
* Try change hidden layer dimension
* Try all friction type and record the best training recipie & Results
* !!! Anomaly detection with the lagrangian !!!





BRANCH 2: 

0) Definition of the problem

A. Complete black box models 
1) Definition of Baseline (nofrict, frict)
2) Definition of Deep (nofrict, frict)
3) Definition of the solver (ClassicODE) 
4) Considerations and problems Baseline vs Deep
    - Friction (with friction it's easier to undarstand the model but also easier to overfit)
    - Activation function (show RELU vs SILU for example)
    - Energy conservation (show drift in time)
    - Error in the model (even if the model is physically wrong the networks have no problem to fit)
    - Evolution of the states in time 
    - Usage of trig layer (with and without maybe)


B. Inserting a structure in the NN (still Black box)
5) Problems of the Hamiltonian in a double pendulum (not usable)
6) Definition of Lagrangian NN
7) Consideration of the LagrangianNN vs BB NN: 
    - evolution in time (much better). Show a comparison
    - evolution of the states (comparable)
    - problems of the hessian which is not pd -> the model in not that stable (activation problems) + needs special activation
    - Error in the model (it cannot converge when there is an error) -> Anomaly detection
    - friction (it cannot converge when there is friction)
    - speed and batch size: it's much slower than a normal BB NN because it doesn't work with many batches (also from the paper)


C. Grey Box LNN 
8) Definition of the Structured LNN (use of Legendre equations using softplus)
9) Consideration Gray box LLN and/vs Black box LNN:
    - hessian is now positive definite -> model is much more stable and can use SILU activation
    - Friction: it's possible to add a friction module so that it can also learn friction (BBLNN is energy conservative while GBLNN is can also be non energy conserving)
    - still it doesn't learn when the model is wrong -> Anomaly detection even though there is friction now
    - Energy of the system that is learned by these models. It stays constant over time (or decrease with friction) it
    is not the same of the normal energy (gauge invariance) -> we should fix something. Possibility to show that V is a multiple of the real V
    - Possibility of the LNN BB to learn any model while the Structured LNN cannot not (only mechanical systems). We should
    add manually modules. 


D. General considerations:
    - Why and how normalization has been implemented and benefits
    - note that rk4 is not sympletic, should use better integrators


Possibility to add
- Try to add a real anomaly detector 
- Make neural ode work + curricular step + horizon (one step ahead vs trajectory training)
- Implement a Neural Probabilistic model to learn a distribution and not only an expected value + 
run Monte Carlo simulations 
- Possibility to learn a particle in a magnetic field with BBLNN and not with GBLNN. 