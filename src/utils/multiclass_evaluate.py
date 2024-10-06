import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

def accuracy(model: torch.nn.Module,
             data: torch.Tensor,
             truth: torch.Tensor,
             classes: int = None) -> float:

    if classes is None:
        classes = torch.unique(truth)

    accuracy_fn = MulticlassAccuracy(classes=classes)

    model.eval()
    with torch.no_grad():
        output = torch.argmax(model(data), dim=1)
        acc = accuracy_fn(output, truth).item()

    return acc



