import numpy as np 
from CostFunction import cost_function
np.set_printoptions(suppress=True)

def gradient_descent(X, y, param, iter, alpha):
    m = X.shape[0]
    J_history = np.zeros((iter,1))
    theta_history = np.zeros((iter,param.shape[0]))
    theta_temp = param.copy()
    theta = param
    for i in range(iter):
        for j in range(theta.shape[0]):
            theta_temp[j] = theta[j] - alpha/m* np.sum((X@theta -y)*X[[j],:])
        theta = theta_temp
        J_history[i] = cost_function(X,y,theta)
        theta_history[[i],:] = np.transpose(theta)

    return theta , J_history, theta_history