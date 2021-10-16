import torch
from torch import nn as nn
from torch import optim as optim
"""
Model based on AlphaZero's architecture
"""


class ResidualLayer(nn.Module):
    """
    Layer with a residual block. 
    Stacked multiple times in our model
    """
    def __init__(self, conv_channels=256):
        super().__init__()
        self.CONV_CHANNELS = conv_channels
        self.conv1 = nn.conv2d(in_channels=self.CONV_CHANNELS, out_channels=self.CONV_CHANNELS, kernel_size=3),
        self.batchnorm1 = nn.BatchNorm2d(num_features=self.CONV_CHANNELS), #TODO
        self.relu = nn.ReLU(),
        self.conv2 = nn.conv2d(in_channels=self.CONV_CHANNELS, out_channels=self.CONV_CHANNELS, kernel_size=3),
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.CONV_CHANNELS), #TODO
 
    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = x + y
        y = self.relu(y)
        return y


class SigmaModel(nn.Module):
    """
    Main model definition
    """
    
    def __init__(self, input_shape, conv_channels : int=256, n_residual_layers : int=40, c=0.0001):
        self.c = c
        self.CONV_CHANNELS = conv_channels
        self.N_RESIDUAL_LAYERS = n_residual_layers
        super(SigmaModel, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.CONV_CHANNELS, kernel_size=3),
            nn.BatchNorm2d(num_features=self.CONV_CHANNELS), #TODO
            nn.ReLU()
        )
        
        self.residual_stack = nn.ModuleList([
            ResidualLayer(self.CONV_CHANNELS) for i in range(self.N_RESIDUAL_LAYERS)
        ])
        
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=self.CONV_CHANNELS, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1), #TODO
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=1), #TODO
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=1), #TODO
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=self.CONV_CHANNELS, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2), #TODO
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=2) #TODO
        )
        

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_stack(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy


# TODO: Custom Loss function that is compatible with backward()? SGD?

class SigmaLoss(nn.Module):
    def __init__(self):
        pass
    
# z is the predicted value, v is the actual value based on mcts
def value_loss(z, v):
    loss = torch.mean((z-v)**2)
    return loss

# pi is the predicted policy, p is the predicted policy based on mcts
def policy_loss(pi, p):
    loss = torch.dot(pi, p)
    return loss

# Add L2 regularization in optimizer instead, not here.

# Loss funciton, excluding L2 reg
def loss(z, v, pi, p):
    return value_loss(z, v) + policy_loss(pi, p)


if __name__ == '__main__':
    # demo/tests

    model = SigmaModel((64, 64), n_residual_layers=10)
    model.train()
    # loss defined above
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=model.c)

    for epoch in range(2):
        for data in []:
            inputs, labels = None, None
            value, policy = model(inputs)
