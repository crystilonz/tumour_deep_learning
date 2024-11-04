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


def evaluate_multiclass_classifier(
        model: torch.nn.Module,
        data: torch.Tensor,
        truth: torch.Tensor,
        classes: int = None,
        conf_mat_normalize: Optional[Literal["true", "pred", "all", "none"]] = "true",
) -> dict:
    """
    :param model: Instance of a model. Must be trained
    :param data: data to input to the model
    :param truth: truth labels of the data
    :param classes: number of classes. If not supplied will check `truth` for unique labels
    :param conf_mat_normalize: how to normalize confusion matrix
    :return: A python Dict of results.
        "top1_acc": macro-averaged top 1 accuracy
        "top3_acc": macro-averaged top 3 accuracy
        "top5_acc": macro-averaged top 5 accuracy
        "confusion_matrix": confusion matrix
        "roc": roc graph in tuples of tensor (fpr, tpr, thresholds)
        "auroc": area under roc curve, macro-averaged over all classes
        "precision": precision over all classes
        "recall": recall over all classes
        "f_one": f_one score, averaged over all classes

        Class specific metrics are returned in Tensor
        "class_top1_acc": top 1 accuracy of each class.
        "class_top3_acc": top 3 accuracy of each class.
        "class_top5_acc": top 5 accuracy of each class.
        "class_precision": precision of each class.
        "class_recall": recall of each class.
        "class_f_one": f one score of each class.
    """
    if classes is None:
        classes = len(torch.unique(truth))

    # accuracies
    top1_acc_fn = MulticlassAccuracy(num_classes=classes, average="macro", top_k=1)
    top3_acc_fn = MulticlassAccuracy(num_classes=classes, average="macro", top_k=3)
    top5_acc_fn = MulticlassAccuracy(num_classes=classes, average="macro", top_k=5)

    class_top1_acc_fn = MulticlassAccuracy(num_classes=classes, average="none", top_k=1)
    class_top3_acc_fn = MulticlassAccuracy(num_classes=classes, average="none", top_k=3)
    class_top5_acc_fn = MulticlassAccuracy(num_classes=classes, average="none", top_k=5)

    # micro
    micro_top1_acc_fn = MulticlassAccuracy(num_classes=classes, average="micro", top_k=1)
    micro_top3_acc_fn = MulticlassAccuracy(num_classes=classes, average="micro", top_k=3)
    micro_top5_acc_fn = MulticlassAccuracy(num_classes=classes, average="micro", top_k=5)

    # auroc
    auroc_fn = MulticlassAUROC(num_classes=classes, average="macro")
    class_auroc_fn = MulticlassAUROC(num_classes=classes, average="none")

    # roc
    roc_fn = MulticlassROC(num_classes=classes)

    # confusion matrix
    conf_matrix_fn = MulticlassConfusionMatrix(num_classes=classes, normalize=conf_mat_normalize)

    # precision
    prec_fn = MulticlassPrecision(num_classes=classes, average="macro")
    class_prec_fn = MulticlassPrecision(num_classes=classes, average="none")
    micro_prec_fn = MulticlassPrecision(num_classes=classes, average="micro")

    # recall
    rec_fn = MulticlassRecall(num_classes=classes, average="macro")
    class_rec_fn = MulticlassRecall(num_classes=classes, average="none")
    micro_rec_fn = MulticlassRecall(num_classes=classes, average="micro")

    # f_one
    f_one = MulticlassFBetaScore(num_classes=classes, beta=1.0, average="macro")
    class_f_one_fn = MulticlassFBetaScore(num_classes=classes, beta=1.0, average="none")
    micro_f_one_fn = MulticlassFBetaScore(num_classes=classes, beta=1.0, average="micro")

    # run through data
    model.eval()
    with torch.no_grad():
        output = model(data)

        # run metrics
        top1_acc = top1_acc_fn(output, truth)
        top3_acc = top3_acc_fn(output, truth)
        top5_acc = top5_acc_fn(output, truth)
        conf_matrix = conf_matrix_fn(output, truth)
        roc = roc_fn(output, truth)
        auroc = auroc_fn(output, truth)
        prec = prec_fn(output, truth)
        rec = rec_fn(output, truth)
        f_one = f_one(output, truth)

        # class
        class_top1_acc = class_top1_acc_fn(output, truth)
        class_top3_acc = class_top3_acc_fn(output, truth)
        class_top5_acc = class_top5_acc_fn(output, truth)
        class_auroc = class_auroc_fn(output, truth)
        class_prec = class_prec_fn(output, truth)
        class_rec = class_rec_fn(output, truth)
        class_f_one = class_f_one_fn(output, truth)

        # micros
        micro_top1_acc = micro_top1_acc_fn(output, truth).item()
        micro_top3_acc = micro_top3_acc_fn(output, truth).item()
        micro_top5_acc = micro_top5_acc_fn(output, truth).item()
        micro_f_one = micro_f_one_fn(output, truth).item()
        micro_prec = micro_prec_fn(output, truth).item()
        micro_rec = micro_rec_fn(output, truth).item()


        # if averaged, then return as a number
        top1_acc = top1_acc.item()
        top3_acc = top3_acc.item()
        top5_acc = top5_acc.item()
        auroc = auroc.item()
        prec = prec.item()
        rec = rec.item()
        f_one = f_one.item()


    # assemble to dict
    return {
        # averaged values
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "confusion_matrix": conf_matrix,
        "roc": roc,
        "auroc": auroc,
        "precision": prec,
        "recall": rec,
        "f_one": f_one,

        # class-specific values
        "class_top1_acc": class_top1_acc,
        "class_top3_acc": class_top3_acc,
        "class_top5_acc": class_top5_acc,
        "class_auroc": class_auroc,
        "class_precision": class_prec,
        "class_recall": class_rec,
        "class_f_one": class_f_one,

        # micros
        "micro_top1_acc": micro_top1_acc,
        "micro_top3_acc": micro_top3_acc,
        "micro_top5_acc": micro_top5_acc
    }







