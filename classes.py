from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import pandas as pd

INNER_WIDTH = 256

class NasdaqDataset(Dataset):
    #
    # Init with a csv file or directly with features & targets
    #
    def __init__(self, csv=None, target_columns=None, features=None, targets=None, device=None):
        if csv:
            self.dataframe = pd.read_csv(csv) 
            features = self.dataframe.iloc[:, :-target_columns].values
            targets = self.dataframe.iloc[:, -target_columns:].values
        if not target_columns:
            target_columns = len(targets[0])
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        # Return the length of the dataset
        return len(self.features)

    def __getitem__(self, idx):
        # Retrieve features and target at the specified index
        return self.features[idx], self.targets[idx]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(60, INNER_WIDTH),
            nn.ReLU(),
            nn.Linear(INNER_WIDTH, INNER_WIDTH),
            nn.ReLU(),
            nn.Linear(INNER_WIDTH, INNER_WIDTH),
            nn.ReLU(),
            nn.Linear(INNER_WIDTH, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
