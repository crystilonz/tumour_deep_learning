import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassROC, MulticlassConfusionMatrix, \
    MulticlassPrecision, MulticlassRecall
from typing import Optional, Literal


def accuracy(model: torch.nn.Module,
             data: torch.Tensor,
             truth: torch.Tensor,
             classes: int = None,
             top_k: int = 1) -> float:

    if classes is None:
        classes = len(torch.unique(truth))

    accuracy_fn = MulticlassAccuracy(num_classes=classes, top_k=top_k)

    model.eval()
    with torch.no_grad():
        output = model(data)
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

def confusion_matrix(model: torch.nn.Module,
                     data: torch.Tensor,
                     truth: torch.Tensor,
                     classes: int = None,
                     normalize: Optional[Literal["true", "pred", "all", "none"]] = "true"):

    if classes is None:
        classes = len(torch.unique(truth))

    conf_fn = MulticlassConfusionMatrix(num_classes=classes, normalize=normalize)
    model.eval()
    with torch.no_grad():
        output = model(data)
        conf_matrix = conf_fn(output, truth)

    return conf_matrix

def precision(model: torch.nn.Module,
              data: torch.Tensor,
              truth: torch.Tensor,
              classes: int = None) -> float:

    if classes is None:
        classes = len(torch.unique(truth))

    prec_fn = MulticlassPrecision(num_classes=classes)
    model.eval()
    with torch.no_grad():
        output = model(data)
        prec = prec_fn(output, truth)

    return prec.item()

def recall(model: torch.nn.Module,
           data: torch.Tensor,
           truth: torch.Tensor,
           classes: int = None) -> float:

    if classes is None:
        classes = len(torch.unique(truth))

    rec_fn = MulticlassRecall(num_classes=classes)
    model.eval()
    with torch.no_grad():
        output = model(data)
        rec = rec_fn(output, truth)

    return rec.item()











