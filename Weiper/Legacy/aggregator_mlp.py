# aggregator_mlp.py
import torch
import torch.nn as nn

class AggregatorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(AggregatorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # outputs a logit

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
