import numpy as np
from Error_functions import *
from Activations import *
from Model import *

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
activation_function = Sigmoid()
error_function = mse

model = TwoLayerPerceptron(2,2,1,activation_function,error_function)
pred_y = model.forward_pass(X)

print(model.gradient_descent(X, y))