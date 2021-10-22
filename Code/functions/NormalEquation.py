import numpy
from numpy.core.fromnumeric import transpose

import numpy as np

def normal_equation(X,y,theta):
    transp = np.transpose(X)
    inverse = np.linalg.inv(transp@X)
    theta = inverse@(transp@y)
    return theta