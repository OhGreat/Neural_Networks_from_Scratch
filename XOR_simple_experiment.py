import numpy as np
import matplotlib.pyplot as plt
from classes.Error_functions import *
from classes.Activations import *
from classes.Model import *

# Set seed for reproducibility
np.random.seed(0)
# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # input
y = np.array([[0],[1],[1],[0]])  # output

# Create Neural Network
input_size = 2
hidden_nodes = 3
output_size = 1
activation_function = Sigmoid()
error_function = MSE()
model = TwoLayerPerceptron( input_size,hidden_nodes,output_size,
                            activation_function,error_function)

# Train the model
loss = model.train(X, y, 0.1, 10000)

# Print results
predictions = model.forward_pass(X).squeeze()
print(f"predictions: {predictions}")
preds_rounded = np.where(predictions > 0.5,1,0)
print(f"predictions rounded: {preds_rounded}")
print(f"expected results: {y.squeeze()}")
# Plot results and save figure
plt.plot(loss)
plt.title(f"2 layer perceptron with {hidden_nodes} nodes in hidden layer")
plt.xlabel("epochs")
plt.ylabel("mean squared error")
plt.savefig('results/loss.png')
