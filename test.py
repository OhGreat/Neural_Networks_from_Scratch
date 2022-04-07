import numpy as np
import matplotlib.pyplot as plt
from Error_functions import mse
from Activations import Sigmoid
from Model import TwoLayerPerceptron


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
activation_function = Sigmoid()
error_function = mse

model = TwoLayerPerceptron(2,2,1,activation_function,error_function)
#pred_y = model.forward_pass(X)

#print(model.gradient_descent(X, y,0.01))

loss = model.train(X, y, 0.1, 10000)
print("predictions:",model.forward_pass(X))

#plt.plot(loss)
#plt.show()

