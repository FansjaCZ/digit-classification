import torch

import utils
from config import data as conf

def train_model(model, data_loader, loss_fn, optimizer):
    model.train()
    device = utils.get_model_device(model)

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_model(model, data_loader, loss_fn):
    model.eval()
    device = utils.get_model_device(model)

    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            total_loss += loss_fn(out, y).item()
            total_correct += (out.argmax(1)==y).type(torch.float).sum().item()

    avg_loss = total_loss/len(data_loader)
    accuracy = 100*(total_correct/len(data_loader.dataset))

    print(f" - LOSS: {round(avg_loss, conf["format"]["loss_decimals"])}\n - ACCURACY: {round(accuracy, conf["format"]["percentage_decimals"])}%")
    return avg_loss, accuracy