# models/mlp.py
import torch.nn as nn  # nn.Linear, BatchNorm, Loss functions
import torch.nn.functional as F  # Functions that don't have any parameters


class MLP(nn.Module):
    def __init__(self, input_size, num_neurons):
        super(MLP, self).__init__()

        # Flatten n x n image into n^2-vector
        self.flatten = nn.Flatten()

        # Layer 1: matrix -> 64 neurons
        self.fc1 = nn.Linear(input_size, num_neurons)
        self.bn1 = nn.BatchNorm1d(num_neurons)

        # Layer 2: Hidden layer to output
        self.fc2 = nn.Linear(num_neurons, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
