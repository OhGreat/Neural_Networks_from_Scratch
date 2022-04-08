from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from classes.Error_functions import *
from classes.Activations import *
from classes.Model import *

# Set seed for reproducibility
np.random.seed(0)
# XOR Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # input
y = np.array([[0],[1],[1],[0]])  # output

input_size = 2
output_size = 1
hidden_nodes = [2, 4, 8, 16]
learning_rates = [0.1]
epochs = [10000]

activation_function = Sigmoid()
error_function = MSE()

losses = []
exp_names = []
for hn in hidden_nodes:
    for lr in learning_rates:
        for epoch in epochs:
            # create model with hn hidden nodes in hidden layer
            model = TwoLayerPerceptron( input_size,hn,output_size,
                            activation_function,error_function)
            # train current model
            loss = model.train(X, y, 0.1, 10000)
            losses.append(loss)

            # evaluate model
            preds = model.forward_pass(X)
            preds = np.where(preds > 0.5,1,0)
            learned = np.array_equal(y,preds)

            # name of current experiment
            exp_name = f"hn_{hn}-lr_{lr}-ok_{learned}"
            exp_names.append(exp_name)

            

# Create plot of comparisons
fig, ax = plt.subplots()
for idx in range(len(losses)):
    ax.plot(losses[idx], label=exp_names[idx])
leg = ax.legend()
plt.title(f"Experiments on NNs")
plt.xlabel("epochs")
plt.ylabel("mean squared error")
plt.savefig('results/losses.png')
