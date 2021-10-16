import torch
from torch import nn as nn

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
        self.batchnorm1 = nn.BatchNorm2d(num_features=self.CONV_CHANNELS),
        self.relu = nn.ReLU(),
        self.conv2 = nn.conv2d(in_channels=self.CONV_CHANNELS, out_channels=self.CONV_CHANNELS, kernel_size=3),
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.CONV_CHANNELS),
 
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

    
    def __init__(self, input_shape, conv_channels : int=256, n_residual_layers : int=40):
        self.CONV_CHANNELS = conv_channels
        self.N_RESIDUAL_LAYERS = n_residual_layers
        super(SigmaModel, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.CONV_CHANNELS, kernel_size=3),
            nn.BatchNorm2d(num_features=self.CONV_CHANNELS),
            nn.ReLU()
        )
        
        self.residual_stack = nn.ModuleList([
            ResidualLayer(self.CONV_CHANNELS) for i in range(self.N_RESIDUAL_LAYERS)
        ])
        
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=self.CONV_CHANNELS, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=1), #TODO
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=1), #TODO
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=self.CONV_CHANNELS, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=2) #TODO
        )
        

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_stack(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

    # TODO: Loss function? SGD?

    