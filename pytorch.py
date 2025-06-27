import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[2.0, 1.0], [0.0, 1.0]])

# Add, Multiply
print("Added:\n", a + b)
print("Multiplied:\n", a * b)

import torch.nn as nn

layer = nn.Linear(3, 2)  # input size = 3, output size = 2

x = torch.tensor([[1.0, 2.0, 3.0]])
output = layer(x)
print("Output:\n", output)

import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
output = model(x)
print("Network output:\n", output)

