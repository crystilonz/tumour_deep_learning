import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
import seaborn as sn
import pandas as pd
from typing import Any, List
import os

ENV_SHOW_PLOT = os.environ.get('ENV_SHOW_PLOT')
if ENV_SHOW_PLOT:
    ENV_SHOW_PLOT = False if ENV_SHOW_PLOT == 'False' else True
else:
    ENV_SHOW_PLOT = True


def plot_loss(training_loss: [float],
              testing_loss: [float],
              save_to: str | Path = None,
              show_plot: bool = True) -> None:
    """Given list of training loss and testing loss on each epoch, plot loss curves
    and save it to a file (optional)"""

    epochs = range(1, len(training_loss) + 1)
    plt.figure(figsize=(7, 7))
    plt.plot(epochs, training_loss, color='blue', label='Training Loss')
    plt.plot(epochs, testing_loss, color='red', label='Testing Loss')
    plt.legend(loc='best')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def plot_roc(false_pos,
             true_pos,
             threshold,
             auroc_by_class: torch.Tensor = None,
             auroc = None,
             title: str = None,
             label_dict: dict[int, str] = None,
             save_to: str | Path = None,
             show_plot: bool = True) -> None:
    """Given a list of fpr, tpr, and threshold, plot ROC curves and save it to
    a file (optional)"""

    plt.figure(figsize=(7, 7))
    for i in range(len(false_pos)):
        label = label_dict[i] if label_dict is not None else f"Label {i}"
        if auroc_by_class is not None and auroc_by_class.numel() == len(false_pos):
            label += f" (AUC = {auroc_by_class[i].item():.05f})"
        plt.plot(false_pos[i], true_pos[i], label=label)
    if title:
        title +=  f" (average AUC = {auroc:.05f})" if auroc else ""
    else:
        title = f"ROC (average AUC = {auroc:.05f})" if auroc else "ROC"
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()

def plot_confusion_matrix(cm: np.ndarray | torch.Tensor,
                          save_to: str | Path = None,
                          indices: List[Any] = None,
                          show_plot: bool = True) -> None:
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()

    if indices is None:
        indices = range(cm.shape[0])

    df_cm = pd.DataFrame(cm, index=indices, columns=indices)
    plt.figure(figsize=(8, 8))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()













