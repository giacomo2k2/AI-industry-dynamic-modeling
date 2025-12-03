import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def energy_metric(real, pred): 
    # R2 score metric
    return r2_score(real, pred)

def evolution_metric(real, pred): 
    return np.sum(np.abs(real - pred)) / real.shape[0]

def main():
    real = np.random.rand(100,4)
    pred = np.random.rand(100,4)
    print(evolution_metric(real, pred))

if __name__ == '__main__':
    main()