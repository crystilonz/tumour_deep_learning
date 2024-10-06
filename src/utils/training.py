from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


def training_step(model: nn.Module,
                  dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  device: torch.device = None) -> (float, float):
    # device agnostic
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loss accumulation
    total_loss = 0

    model.to(device)  # put model on device
    model.train()

    for data, label in dataloader:
        data, label = data.to(device), label.to(device)  # put data on device

        # forward pass
        prediction_probs = model(data)

        # calculate loss, acc
        loss = loss_fn(prediction_probs, label)
        total_loss += loss.item()

        optimizer.zero_grad()

        # backward, step
        loss.backward()
        optimizer.step()

    # average
    total_loss /= len(dataloader)

    return total_loss


def testing_step(model: nn.Module,
              dataloader: DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = None) -> float:

    # device agnostic
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loss accumulation
    total_loss = 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)  # move to device

            prediction_probs = model(data)

            # calculate loss
            loss = loss_fn(prediction_probs, label)
            total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss


def train_model(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                epochs: int,
                device: torch.device = None) -> ([float], [float]):

    training_losses = []
    testing_losses = []

    for epoch in tqdm(range(epochs)):
        training_loss = training_step(model, train_dataloader, optimizer, loss_fn, device)
        testing_loss = testing_step(model, test_dataloader, loss_fn, device)
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        if epoch % 10 == 0 and epoch > 0:
            # report
            print(f"Epoch: {epoch} >>> Training Loss: {training_loss} | Testing Loss: {testing_loss}")

    print("------------------------------")
    print(f"After training >>> Training Loss: {training_losses[-1]} | Testing Loss: {testing_losses[-1]}")

    return training_losses, testing_losses


