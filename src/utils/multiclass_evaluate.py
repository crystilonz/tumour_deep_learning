import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassROC


def accuracy(model: torch.nn.Module,
             data: torch.Tensor,
             truth: torch.Tensor,
             classes: int = None) -> float:

    if classes is None:
        classes = len(torch.unique(truth))

    accuracy_fn = MulticlassAccuracy(num_classes=classes)

    model.eval()
    with torch.no_grad():
        output = torch.argmax(model(data), dim=1)
        acc = accuracy_fn(output, truth).item()

    return acc

def auroc(model: torch.nn.Module,
          data: torch.Tensor,
          truth: torch.Tensor,
          classes: int = None):

    # one vs rest AUROC

    if classes is None:
        classes = len(torch.unique(truth))

    auroc_fn = MulticlassAUROC(num_classes=classes)
    model.eval()
    with torch.no_grad():
        output = model(data)
        auroc = auroc_fn(output, truth).item()

    return auroc

def roc(model: torch.nn.Module,
        data: torch.Tensor,
        truth: torch.Tensor,
        classes: int = None):

    if classes is None:
        classes = len(torch.unique(truth))

    roc_fn = MulticlassROC(num_classes=classes)
    model.eval()

    with torch.no_grad():
        output = model(data)
        fpr, tpr, thresholds = roc_fn(output, truth)

    return fpr, tpr, thresholds








