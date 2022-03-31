import numpy as np

class Activation():
    def __call__(x):
        pass

    def derivative(x):
        pass

class Sigmoid(Activation):

    def __call__(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)
