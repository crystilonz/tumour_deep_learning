import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassROC, MulticlassConfusionMatrix, \
    MulticlassPrecision, MulticlassRecall, MulticlassFBetaScore
from typing import Optional, Literal


def accuracy(model: torch.nn.Module,
             data: torch.Tensor,
             truth: torch.Tensor,
             classes: int = None,
             top_k: int = 1,
             average: Literal["none", "macro", "weighted"] | None = "macro") -> float | torch.FloatTensor:

    if classes is None:
        classes = len(torch.unique(truth))

    accuracy_fn = MulticlassAccuracy(num_classes=classes, top_k=top_k, average=average)

    model.eval()
    with torch.no_grad():
        output = model(data)
        acc = accuracy_fn(output, truth)

        if acc.numel() == 1:
            acc = acc.item()

    return acc

def auroc(model: torch.nn.Module,
          data: torch.Tensor,
          truth: torch.Tensor,
          classes: int = None,
          average: Literal["none", "macro", "weighted"] | None = "macro") -> float | torch.FloatTensor:

    # one vs rest AUROC, averaged by class

    if classes is None:
        classes = len(torch.unique(truth))

    auroc_fn = MulticlassAUROC(num_classes=classes, average=average)
    model.eval()
    with torch.no_grad():
        output = model(data)
        auroc = auroc_fn(output, truth)

        if auroc.numel() == 1:
            auroc = auroc.item()

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
              classes: int = None,
              average: Literal["none", "macro", "weighted"] | None = "macro") -> float | torch.FloatTensor:

    if classes is None:
        classes = len(torch.unique(truth))

    prec_fn = MulticlassPrecision(num_classes=classes, average=average)
    model.eval()
    with torch.no_grad():
        output = model(data)
        prec = prec_fn(output, truth)

        if prec.numel() == 1:
            prec = prec.item()

    return prec

def recall(model: torch.nn.Module,
           data: torch.Tensor,
           truth: torch.Tensor,
           classes: int = None,
           average: Literal["none", "macro", "weighted"] | None = "macro") -> float | torch.FloatTensor:

    if classes is None:
        classes = len(torch.unique(truth))

    rec_fn = MulticlassRecall(num_classes=classes, average=average)
    model.eval()
    with torch.no_grad():
        output = model(data)
        rec = rec_fn(output, truth)

        if rec.numel() == 1:
            rec = rec.item()

    return rec


def f_beta(model: torch.nn.Module,
       data: torch.Tensor,
       truth: torch.Tensor,
       classes: int = None,
       beta: float = 1.0,
       average: Literal["none", "macro", "weighted"] | None = "macro") -> float | torch.FloatTensor:

    if classes is None:
        classes = len(torch.unique(truth))

    fbeta_fn = MulticlassFBetaScore(num_classes=classes, beta=beta, average=average)
    model.eval()
    with torch.no_grad():
        output = model(data)
        fbeta = fbeta_fn(output, truth)

        if fbeta.numel() == 1:
            fbeta = fbeta.item()

    return fbeta











