import numpy as np

def evolution_metric(gt, pred):    
    return np.sum(np.abs(gt - pred)) / (gt.shape[0] * gt.shape[1])
