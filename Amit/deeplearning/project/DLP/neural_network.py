import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.activation = activation
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x