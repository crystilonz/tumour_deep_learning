from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from pathlib import Path

from models.interface.LungRNN import LungRNN


def training_step(model: nn.Module,
                  dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  device: torch.device = None) -> float:
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

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f"Training model: {model.__class__.__name__} for {epochs} epochs.")

    for epoch in tqdm(range(epochs)):
        training_loss = training_step(model, train_dataloader, optimizer, loss_fn, device)
        testing_loss = testing_step(model, test_dataloader, loss_fn, device)
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        # if epoch % 10 == 0 and epoch > 0:
        #     # report
        #     print(f"Epoch: {epoch} >>> Training Loss: {training_loss} | Testing Loss: {testing_loss}")

    # print("------------------------------")
    print(f"After training >>> Training Loss: {training_losses[-1]} | Testing Loss: {testing_losses[-1]}")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return training_losses, testing_losses


def save_model(model: nn.Module,
               save_path: str | Path):
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    if save_path.suffix != '.pt' and save_path.suffix != '.pth':
        save_path = save_path.parent / (save_path.stem + '.pt')

    torch.save(obj=model.state_dict(),
               f=save_path)


def rnn_training_step(model: LungRNN,
                      dataloader: DataLoader,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: torch.nn.Module,
                      device: torch.device = None):
    # device agnostic
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # total loss accumulation
    total_loss = 0
    model.to(device)
    model.train()

    for feature, caption in dataloader:
        feature, caption = feature.to(device), caption.to(device)

        cap_preds = model(feature, caption)

        # calculate loss
        batch_vocab_dim = torch.transpose(cap_preds, 1, 2)
        loss = loss_fn(batch_vocab_dim, caption)
        total_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    # average
    total_loss /= len(dataloader)
    return total_loss


def rnn_testing_step(model: LungRNN,
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
        for feature, caption in dataloader:
            feature, caption = feature.to(device), caption.to(device)  # move to device

            cap_preds = model(feature, caption)

            # calculate loss
            batch_vocab_dim = torch.transpose(cap_preds, 1, 2)
            loss = loss_fn(batch_vocab_dim, caption)
            total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss


def rnn_train_model(model: LungRNN,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module,
                    train_dataloader: DataLoader,
                    test_dataloader: DataLoader,
                    epochs: int,
                    device: torch.device = None) -> ([float], [float]):
    training_losses = []
    testing_losses = []
    ten_epochs = 0

    # taking a sample from test data_loader
    train_show_feature, train_show_caption = next(iter(train_dataloader))
    train_show_feature = train_show_feature[0]
    train_show_caption = train_show_caption[0]
    test_show_feature, test_show_caption = next(iter(test_dataloader))
    test_show_feature = test_show_feature[0]
    test_show_caption = test_show_caption[0]

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f"Training model: {model.__class__.__name__} for {epochs} epochs.")

    for epoch in tqdm(range(epochs)):
        training_loss = rnn_training_step(model, train_dataloader, optimizer, loss_fn, device)
        testing_loss = rnn_testing_step(model, test_dataloader, loss_fn, device)
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        if (epoch + 1) / epochs * 10 > ten_epochs:
            ten_epochs += 1

            # run captioning on the image
            train_pred_caption = model.caption(train_show_feature, 50)
            test_pred_caption = model.caption(test_show_feature, 50)

            # print
            print(f"\nEpoch {epoch}")
            print(f"Training Loss: {training_loss}")
            print("Training Sample:")
            print("    Prediction:", train_pred_caption)
            print("    Actual:", model.vocab.translate_from_index_list(train_show_caption))
            print(f"Testing Loss: {testing_loss}")
            print("Testing Sample:")
            print("    Prediction:", test_pred_caption)
            print("    Actual:", model.vocab.translate_from_index_list(test_show_caption), '\n')


    print(f"After training >>> Training Loss: {training_losses[-1]} | Testing Loss: {testing_losses[-1]}")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return training_losses, testing_losses
