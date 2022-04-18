import torch.nn as nn
import torch


class LinearRegression(nn.Module):
    """
    Linear Regression Model
    """
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
