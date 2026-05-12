# models/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1: 1 channel -> 8 filters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Layer 2: 8 -> 16 filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: Detects edges/textures
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces image size by half

        # Layer 2: Combines edges into patterns
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces size again

        # If your input image is 28x28, after two pools it is 7x7
        # 16 filters * 7 * 7 = 784
        self.fc1 = nn.Linear(16 * 8 * 8, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


class MaskedSimpleCNN(nn.Module):
    def __init__(self):
        super(MaskedSimpleCNN, self).__init__()
        # Layer 1: Detects edges/textures
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces image size by half

        # Layer 2: Combines edges into patterns
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces size again

        # If your input image is 28x28, after two pools it is 7x7
        # 16 filters * 7 * 7 = 784
        self.fc1 = nn.Linear(16 * 8 * 8, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


class MaskedCNN(nn.Module):
    def __init__(self):
        super(MaskedCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)  # Helps with noisy/masked inputs
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 19 * 19, 1) # For 77 x 77
        #self.fc1 = nn.Linear(16 * 8 * 8, 1) # For 32 x 32

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # Direction 2: Lower capacity (4 and 8 channels instead of 8 and 16)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)

        # GAP: collapses 19x19 spatial grid into 1x1 value per channel. Care about 'what' was seen, not 'where' it was.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Now the FC layer only has 8 inputs (one per feature map)
        self.fc1 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)

class SuperSmallCNN(nn.Module):
    def __init__(self):
        super(SuperSmallCNN, self).__init__()
        # Only 1 filter: can only detect one type of feature
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        # Flattening instead of GAP: forces spatial dependency
        # Input 77x77 -> after 3x3 conv -> 77x77 (due to padding)
        self.fc1 = nn.Linear(77 * 77, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        return self.fc1(x)
    
class ModeratelySmallCNN(nn.Module):
    def __init__(self):
        super(ModeratelySmallCNN, self).__init__()
        # Increased to 4 filters to capture vertical, horizontal, and patchy variations
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        # AdaptiveAvgPool2d(1) acts as Global Average Pooling in PyTorch
        self.gap = nn.AdaptiveAvgPool2d(1)
        # The linear layer now maps the 4 global features to the 1 output class
        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # GAP reduces the 77x77 spatial dimensions to 1x1
        x = self.gap(x)
        # Flatten the remaining 1x1 spatial dimensions: (Batch, Channels, 1, 1) -> (Batch, Channels)
        x = torch.flatten(x, 1)
        return self.fc1(x)