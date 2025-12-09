import numpy as np

def energy_evolution_metric(real, pred):    
    return np.sum(np.square(real - pred)) / (real.shape[0] * real.shape[1])

    