# Neural Network from scratch
Small library of neural networks created from scratch with python and numpy. Created to learn how gradient descent and backpropagation work in order to train a model.
The networks are then tested by teaching them to emulate the XOR logic gate.

## Installation

Install python 3 and numpy >=1.19

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
It interesting to note that when running the network with only two hidden nodes (which is the minimum requirement to emulate the XOR logic gate), with seed set to 0, the network fails to learn to simulate XOR. This is the result of bad weight initialization and the more neurons we add to the hidden layer, the easier it becomes for the network to learn and the harder it is to fall into a local minima that will prevent our network to learn further.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
