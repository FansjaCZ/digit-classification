import torch
from torch import nn
import math

class Model(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        post_conv_im_size = (((im_size[0]-2)/2-2)//2, ((im_size[1]-2)/2-2)//2)
        self.stack = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(int(16*post_conv_im_size[0]*post_conv_im_size[1]), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, inp):
        return self.stack(inp)