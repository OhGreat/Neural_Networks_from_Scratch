import numpy as np
import matplotlib.pyplot as plt
from Error_functions import mse
from Activations import *
from Model import TwoLayerPerceptron

#np.random.seed(0)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
activation_function = Sigmoid()
error_function = mse

model = TwoLayerPerceptron(2,2,1,activation_function,error_function)
#pred_y = model.forward_pass(X)

#print(model.gradient_descent(X, y,0.01))

"""print("predictions:",model.forward_pass(X))
print("hidden weights:",model.hidden_weights)
print("output weights:",model.output_weights)"""
loss = model.train(X, y, 0.1, 10000)
print("predictions:",model.forward_pass(X))
#print("hidden weights:",model.hidden_weights)
#print("output weights:",model.output_weights)

plt.plot(loss)
plt.show()

