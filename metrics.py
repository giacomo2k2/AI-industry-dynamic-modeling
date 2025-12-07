import numpy as np


def state_evolution_metric(real, pred): 
    return np.sum(np.abs(real - pred)) / real.shape[0]


