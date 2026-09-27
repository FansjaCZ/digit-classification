import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import v2

import numpy
from pathlib import Path
from matplotlib import pyplot as plt

import network
import train
from config import data as conf

im_size = conf["main"]["image_size"]
device = conf["main"]["device"]

batch_size = conf["train"]["batch_size"]
epochs = conf["train"]["epochs"]
learning_rate = conf["train"]["learning_rate"]

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

model = network.Model(im_size)
model.to(device)

train_data_loader = DataLoader(train_data, batch_size, True)
test_data_loader = DataLoader(test_data, batch_size, True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), learning_rate)

avg_losses = []
accuracies = []
epochs_list = range(1, epochs+1)
for i in range(epochs):
    train.train_model(model, train_data_loader, loss_fn, optimizer)
    print(f"------------------------\nEpoch {str(i+1)}")
    avg_loss, accuracy = train.test_model(model, test_data_loader, loss_fn)
    avg_losses.append(avg_loss)
    accuracies.append(accuracy)
print(f"------------------------\nCompleted!")

plt.figure()

plt.subplot(1,2,1)
plt.plot(epochs_list, avg_losses)
plt.ylim(0, max(avg_losses))
plt.grid(True)
plt.title("Loss over training")
plt.ylabel("avarage loss")
plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.plot(epochs_list, accuracies)
plt.ylim(0, 100)
plt.grid(True)
plt.title("Accuracy over training")
plt.ylabel("avarage accuracy")
plt.xlabel("epoch")

plt.show()