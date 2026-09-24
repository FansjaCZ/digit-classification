import torch
from torchvision import datasets
from torchvision.transforms import v2
# import network
from pathlib import Path

im_size = (28, 28)

train_data = datasets.MNIST(
    root=Path(__file__).parent.parent / "data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.MNIST(
    root=Path(__file__).parent.parent / "data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

# model = network(im_size)

