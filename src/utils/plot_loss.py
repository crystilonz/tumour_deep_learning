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






