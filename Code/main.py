"""import numpy as np
import pandas as pd
from functions.LoadData import load_data, getX_y
from functions.CostFunction import cost_function
from functions.GradientDescent import gradient_descent
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

data = load_data('./Data/audi.csv')

X, y = getX_y(data, 'price')
colnames = X.columns
tarname = y.name

# Matrix creation 
X = X.to_numpy()
one = np.ones((X.shape[0],1))
X  =np.append(one, X, axis=1)
y = y.to_numpy().reshape((X.shape[0],1))
theta = np.zeros((X.shape[1],1))

# Compute cost function
J = cost_function(X, y, theta)

#plt.plot(np.arange(0,ITER,1),J_history)
#plt.show()"""