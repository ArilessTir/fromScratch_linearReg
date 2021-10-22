from os import error


import numpy as np
def cost_function(X, y ,theta):
    m = X.shape[0]
    X_theta = X.dot(theta)
    diff = X_theta - y
    sum_squared = (np.sum(diff))**2
    J = sum_squared/(2*m)
    return J