import numpy as np
from numpy.lib.index_tricks import AxisConcatenator
import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    return data

def getX_y(data, target):
    X = data.drop(target, axis = 1)
    y = data[target]
    return X, y