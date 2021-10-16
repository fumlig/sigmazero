# Deep reinforcement learning models and algorithms

The models and algorithms implemented are heavily based on the work related to Deep Mind's AlphaZero chess engine. 

ELF OpenGo: https://arxiv.org/pdf/1902.04522.pdf

## Our version of the AlphaZero model

The AlphaZero architecture is heavily based on the AlphaGo Zero architecture which is briefly described below. Sources: https://www.youtube.com/watch?v=OPgRNY3FaxA, 

### Brief intro to layers used:
* BatchNorm: Normalize weights into same range ([0, 1]). Reduces overfitting among other benefits
* Residual Blocks: Pass raw input from previous layer to next layer. Mitigate diminishing gradients issue

## Network architecture

### Input layer
### Convolutional layer: 
- 256 channels (filters) with kernel size 3x3
- BatchNorm
- ReLU
### Residual Layer x 40: (change 40 to lower size for cost reduction)
- 256 channel convolutional layer (3x3 kernel size)
- BatchNorm
- ReLU
- 256 channel convolutional layer (3x3 kernel size)
- BatchNorm
- Residual Block (from raw input to the residual layer)
- ReLU
### Output layer
The network is then split into two output heads:
#### Value Head

outputs value of current state ("probability")

- 1 channel convolutional layer (1x1 kernel size)
- BatchNorm
- ReLU
- Fully Connected Layer with 256 neurons
- ReLU
- Fully connected layer
- tanh activation function

#### Policy head

Outputs policy

- 2 channel convolutional layer (1x1 kernel size)
- BatchNorm
- ReLU
- Fully connected layer. This gives logits for each possible move

## Training algorithm

Gradient descent.
Loss function = value loss + policy loss + regularization

value loss = MSE between the value predicted by the network and the actual value calculated by the monte carlo tree search.

Policy loss = cross entropy between move probabilities (logits) calculated by the network and the monte carlo tree search

Regularization: (similar to Lasso regression) punishes large weight values. L2 weight regularization. Equation

$$ c*||\theta||^2 $$

, or sum of squares of all weights. c=0.0001.

All and all, this results in the following loss function

$$ L = (z-v)^2 - \pi*log(p) + c*||\theta ||^2 $$

Stochastic Gradient Descent is used, with a momentum optimizer. Momentum is set to 0.9. The learning rate goes from  0.01 to 0.0001