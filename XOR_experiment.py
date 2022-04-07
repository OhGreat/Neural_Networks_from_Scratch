import numpy as np
import matplotlib.pyplot as plt
from Error_functions import mse
from Activations import *
from Model import *

# Set seed for reproducibility
np.random.seed(0)
# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

activation_function = Sigmoid()
error_function = mse
model = TwoLayerPerceptron(2,2,1,activation_function,error_function)

# Train the model
loss = model.train(X, y, 0.1, 10000)

# Show results
print("predictions:",model.forward_pass(X))
plt.plot(loss)
plt.show()

