# Neural Network from scratch
Small library of neural networks created from scratch with python and numpy.
Created to learn in depth about gradient descent and backpropagation.
The networks are then tested by teaching them to emulate the XOR logic gate.

## Installation
Python 3 and numpy >=1.19 are required

## Usage
To run a single experiment use the file XOR_experiment.py
``` 
python XOR_experiment.py
```

To run a multiple experiments and get a plot of the results use the file experiments.py
``` 
python experiments.py
```

## Observations
It interesting to note that when running the network with only two hidden nodes (which is the minimum requirement to emulate the XOR logic gate), the network sometimes fails to learn to simulate XOR. An example of this behaviour can be seen when running experiments with np.random.seed(0). In general, this is the result of bad weight initialization and the more neurons we add to the hidden layer, the harder it is for weight inintialization to influence the learning of the model.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
