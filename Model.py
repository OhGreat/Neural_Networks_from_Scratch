from matplotlib.pyplot import axes
import numpy as np


class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, activation, error_function):
        self.hidden_weights = np.random.normal(size=(input_size, hidden_size))
        self.hidden_bias = np.random.uniform(size=(1, hidden_size))

        self.output_weights = np.random.normal(size=(hidden_size, output_size))
        self.output_bias = np.random.uniform(size=(1, output_size))

        self.activation = activation
        self.error_function = error_function

    def forward_pass(self, input, return_activations=False):
        """ The forward pass of the neural network.
            Parameters:
                - input: variables to pass to the network
                - return_activations: returns activations of hidden and output layer 
        """
        hidden_layer_out = np.dot(input, self.hidden_weights) + self.hidden_bias
        hidden_layer_out = self.activation(hidden_layer_out)

        output_layer_out = np.dot(hidden_layer_out, self.output_weights) + self.output_bias
        output_layer_out = self.activation(output_layer_out)

        if return_activations:
            return hidden_layer_out, output_layer_out
        return output_layer_out

    def gradient_descent(self, X, y_true, lr):
        """ Gradient descent step
        """
        # take each layer output value
        hidden_layer_out, output_layer_out = self.forward_pass(X,return_activations=True)
        # Propagate error to output layer
        errors = y_true - output_layer_out
        d_predicted_output = errors * self.activation.derivative(output_layer_out)
        #Propagate error to hidden layer
        error_hidden_layer = np.dot(d_predicted_output,self.output_weights.T)
        d_hidden_layer = error_hidden_layer * self.activation.derivative(hidden_layer_out)
        # Update weights and biases
        self.output_weights += np.dot(hidden_layer_out.T,d_predicted_output) * lr
        self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        self.hidden_weights += np.dot(X.T,d_hidden_layer) * lr
        self.hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
        # Return mean square error
        mse_error = self.error_function(y_true, output_layer_out,ax=0)
        return mse_error

    def train(self, X, y, lr, epochs):
        loss = []
        for i in range(epochs):
            loss.append(self.gradient_descent(X, y, lr))
        return loss
