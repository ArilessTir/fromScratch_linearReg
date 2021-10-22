import numpy as np
from numpy import ma

def min_max_scaler(data):
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    data_norm = (data - min) / (max - min)
    return data_norm