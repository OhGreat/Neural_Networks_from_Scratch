# Neural Network from scratch
![Screenshot](net.png) <br/><br/>
Small library of neural networks created from scratch with python and numpy. <br/>
This project was created during my Master's course at Leiden University. Its purpose was to learn in depth about gradient descent and backpropagation. <br/>
The architecture allows to create a two layer network with an arbitrary number of inputs, hidden nodes in the hidden layer and output nodes.<br/>
The networks have been tested by teaching them to emulate the XOR logic gate.

## Installation
Python 3 and numpy >=1.19 are required.

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
It interesting to note that when running the network with only two hidden nodes (which is the minimum requirement to emulate the XOR logic gate), the network sometimes fails to learn to simulate XOR. <br/>
An example of this behaviour can be seen when running experiments with the seed set as <b>np.random.seed(0)</b>. In general, this is the result of bad weight initialization. The more neurons we add to the hidden layer, the harder it is for weight inintialization to influence the learning of the model.<br/>
An example of the behaviour explained above can be seen on the image below:

![Screenshot](results/losses.png)

The labels represent the parameters of the experiments:<br/>
<ul>
  <li><b>hn</b> : denotes the hidden nodels in the hidden layer of the network.</li>
  <li><b>lr</b> : denotes the learning rate used during training. </li>
  <li><b>working</b>: is a boolean value representing if the network learnt to simulate the XOR logic gate properly.</li>
</ul>

It is obvious that the experiment with two hidden nodes was not able to learn the XOR logic gate properly, converging to a  Mean Squared Error of around 0.15.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Future Work
- Implement multiple layers in the network.

## References 
<b>Deep Learning</b> by Aaron Courville, Ian Goodfellow, and Yoshua Bengio
<br/>
<b>Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems</b> by Geron Aurelien


