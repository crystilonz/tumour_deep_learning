import matplotlib.pyplot as plt
from pathlib import Path

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

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot:
        plt.show()

    plt.close()


def plot_roc(false_pos,
             true_pos,
             threshold,
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
        plt.plot(false_pos[i], true_pos[i], label=label)
    if title:
        title +=  f" (AUC = {auroc:.05f})" if auroc else ""
    else:
        title = f"ROC (AUC = {auroc:.05f})" if auroc else "ROC"
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot:
        plt.show()

    plt.close()




