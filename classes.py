from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import pandas as pd

INNER_WIDTH = 256

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17, INNER_WIDTH),
            nn.ReLU(),
            nn.Linear(INNER_WIDTH, INNER_WIDTH),
            nn.ReLU(),
            nn.Linear(INNER_WIDTH, 6),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
