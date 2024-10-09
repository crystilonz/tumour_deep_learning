from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np

from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_DICT
from models.pancancer_classifier import PanCancerClassifier
from utils.datadump import save_to_json
from utils.multiclass_evaluate import accuracy, auroc, roc
from utils.plotting import plot_loss, plot_roc
from utils.training import train_model

DEFAULT_DATA_DIR = Path(__file__).parent / "datasets" / "pancancer_WSI_representation"
DEFAULT_SAVED_MODEL_PARENT = Path(__file__).parent / "saved_models"
DEFAULT_SAVED_MODEL_DIR_NAME = "pancancer_classifier"
DEFAULT_SAVED_MODEL_NAME = "pancancer_classifier_checkpoints"
DEFAULT_PLOT_NAME = "loss_curve"
DEFAULT_TRAINING_ROC = "train_roc"
DEFAULT_TEST_ROC = "test_roc"
DEFAULT_SAVED_METRIC_NAME = "metrics.json"
DEFAULT_EPOCH = 30
DEFAULT_HIDDEN_SIZE = 42
DEFAULT_BATCH_SIZE = 64


def train_pan_cancer(pan_cancer_model: nn.Module,
                     datadir: str = DEFAULT_DATA_DIR,
                     epochs: int = DEFAULT_EPOCH,
                     batch_size: int = DEFAULT_BATCH_SIZE,
                     save_parent: str | Path = DEFAULT_SAVED_MODEL_PARENT,
                     save_dir: str | Path = DEFAULT_SAVED_MODEL_DIR_NAME,
                     save_name: str | Path = DEFAULT_SAVED_MODEL_NAME,
                     loss_plot_name: str | Path = DEFAULT_PLOT_NAME,
                     train_roc_name: str | Path = DEFAULT_TRAINING_ROC,
                     test_roc_name: str | Path = DEFAULT_TEST_ROC,
                     metric_name: str | Path = DEFAULT_SAVED_METRIC_NAME) -> torch.nn.Module:
    samples, slides, data, labels = get_pancancer_data_from_csv(datadir)
    training_data, testing_data, training_labels, testing_labels = train_test_split(data,
                                                                                    labels,
                                                                                    test_size=0.2,
                                                                                    shuffle=True)
    # change from np.array to tensors
    training_data_tensor = torch.FloatTensor(training_data)
    testing_data_tensor = torch.FloatTensor(testing_data)
    training_labels_tensor = torch.LongTensor(training_labels)
    testing_labels_tensor = torch.LongTensor(testing_labels)

    # change tensors to torch dataset
    train_dataset = TensorDataset(training_data_tensor, training_labels_tensor)
    test_dataset = TensorDataset(testing_data_tensor, testing_labels_tensor)

    # put in dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

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
    save_dir = save_parent / save_dir
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # save model
    torch.save(pan_cancer_model.state_dict(), save_dir / save_name)

    # plot loss curves
    plot_loss(training_losses, testing_losses,
              save_to=save_dir / loss_plot_name,
              show_plot=True)

    # evaluate model
    final_train_acc = accuracy(model=pan_cancer_model,
                               data=training_data_tensor,
                               truth=training_labels_tensor,
                               classes=10)
    final_test_acc = accuracy(model=pan_cancer_model,
                              data=testing_data_tensor,
                              truth=testing_labels_tensor,
                              classes=10)
    final_train_area = auroc(model=pan_cancer_model,
                             data=training_data_tensor,
                             truth=training_labels_tensor,
                             classes=10)
    final_test_area = auroc(model=pan_cancer_model,
                            data=testing_data_tensor,
                            truth=testing_labels_tensor,
                            classes=10)
    train_fpr, train_tpr, train_threshold = roc(model=pan_cancer_model,
                                                data=training_data_tensor,
                                                truth=training_labels_tensor,
                                                classes=10)
    test_fpr, test_tpr, test_threshold = roc(model=pan_cancer_model,
                                             data=testing_data_tensor,
                                             truth=testing_labels_tensor,
                                             classes=10)
    # report metrics
    print(f"Training Dataset")
    print(f"Accuracy: {final_train_acc:.2f}%")
    print(f"AUROC: {final_train_area:.5f}")
    print("------------------------------------")
    print(f"Testing Dataset")
    print(f"Accuracy: {final_test_acc:.2f}%")
    print(f"AUROC: {final_test_area:.5f}")

    # compile metric
    metrics = {"train_acc": final_train_acc,
               "train_area": final_train_area,
               "test_acc": final_test_acc,
               "test_area": final_test_area}

    save_to_json(metrics, save_dir / metric_name)

    # plot ROC curves and save it
    plot_roc(train_fpr, train_tpr, train_threshold,
             auroc=final_train_area,
             title="Training ROC",
             label_dict=PAN_CANCER_DICT,
             save_to=save_dir / train_roc_name)

    plot_roc(test_fpr, test_tpr, test_threshold,
             auroc=final_test_area,
             title="Testing ROC",
             label_dict=PAN_CANCER_DICT,
             save_to=save_dir / test_roc_name)

    return pan_cancer_model


if __name__ == "__main__":
    model = PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)
    train_pan_cancer(pan_cancer_model=model,
                     epochs=40)
