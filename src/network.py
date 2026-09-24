import torch
from torch import nn

class NeuralNetwork(torch.Module):
    def __init__(self, im_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(im_size[0]*im_size[1], 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
        )
    def forward(self, inp):
        flat_inp = self.flatten(inp)
        return self.stack(flat_inp)