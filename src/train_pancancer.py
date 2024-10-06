from data_manipulation.pancancer_from_csv import get_data_from_csv
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from models.pancancer_classifier import PanCancerClassifier
from utils.plot_loss import plot_loss
from utils.training import train_model

DEFAULT_DATA_DIR = Path(__file__).parent / "datasets" / "pancancer_WSI_representation"
DEFAULT_SAVED_MODEL_DIR = Path(__file__).parent / "saved_models" / "pancancer_classifier"
DEFAULT_SAVED_MODEL_NAME = "pancancer_classifier"
DEFAULT_PLOT_NAME = "loss_curve"
DEFAULT_EPOCH = 30
DEFAULT_HIDDEN_SIZE = 42
DEFAULT_BATCH_SIZE = 64


def train_panCancer(datadir: str = DEFAULT_DATA_DIR,
                    epochs: int = DEFAULT_EPOCH,
                    hidden_size: int = DEFAULT_HIDDEN_SIZE,
                    save_dir: str | Path = DEFAULT_SAVED_MODEL_DIR) -> torch.nn.Module:

    samples, slides, data, labels = get_data_from_csv(datadir)
    training_data, testing_data, training_labels, testing_labels = train_test_split(data,
                                                                                    labels,
                                                                                    test_size=0.2,
                                                                                    shuffle=True)
    train_dataset = TensorDataset(torch.FloatTensor(training_data), torch.LongTensor(training_labels))
    test_dataset = TensorDataset(torch.FloatTensor(testing_data), torch.LongTensor(testing_labels))

    train_dataloader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=DEFAULT_BATCH_SIZE)

    # initiate model
    pan_cancer_model = PanCancerClassifier(input_size=len(training_data[0]),
                                           hidden_size=hidden_size,
                                           output_size=len(training_labels))

    # loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pan_cancer_model.parameters(),
                                 lr=0.001)

    training_losses, testing_losses = train_model(model=pan_cancer_model,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  epochs=epochs,
                                                  train_dataloader=train_dataloader,
                                                  test_dataloader=test_dataloader)

    # save plot and model
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # save model
    torch.save(pan_cancer_model.state_dict(), save_dir / DEFAULT_SAVED_MODEL_NAME)

    # plot loss curves
    plot_loss(training_losses, testing_losses,
              save_to=save_dir / DEFAULT_PLOT_NAME,
              show_plot=True)

    return pan_cancer_model


if __name__ == "__main__":
    train_panCancer(epochs=50)

