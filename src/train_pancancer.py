from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np

from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_DICT, PAN_CANCER_LABELS
from models.pan_cancer_classifier import PanCancerClassifier
from utils.datadump import save_to_json
from utils.multiclass_evaluate import accuracy, auroc, roc, confusion_matrix, recall, precision, f_beta, \
    evaluate_multiclass_classifier
from utils.plotting import plot_loss, plot_roc, plot_confusion_matrix
from utils.training import train_model, save_model

DEFAULT_DATA_DIR = Path(__file__).parent / "datasets" / "pancancer_WSI_representation"
DEFAULT_SAVED_MODEL_PARENT = Path(__file__).parent / "saved_models"
DEFAULT_SAVED_MODEL_DIR_NAME = "pancancer_classifier"
DEFAULT_SAVED_MODEL_NAME = "pancancer_classifier_checkpoints"
DEFAULT_PLOT_NAME = "loss_curve"
DEFAULT_TRAINING_ROC = "train_roc"
DEFAULT_TEST_ROC = "test_roc"
DEFAULT_SAVED_METRIC_NAME = "metrics.json"
DEFAULT_CONFUSION_MATRIX_NAME = "confusion_matrix"
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
                     metric_name: str | Path = DEFAULT_SAVED_METRIC_NAME,
                     conf_mat_name: str | Path = DEFAULT_CONFUSION_MATRIX_NAME) -> torch.nn.Module:

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

    # loss weights on training
    total_label = len(training_labels)
    weights = [total_label/(np.sum(training_labels == i) * 10) for i in range(10)]

    # loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    optimizer = torch.optim.Adam(pan_cancer_model.parameters(),
                                 lr=0.001)

    training_losses, testing_losses = train_model(model=pan_cancer_model,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  epochs=epochs,
                                                  train_dataloader=train_dataloader,
                                                  test_dataloader=test_dataloader)
    save_dir = save_parent / save_dir
    # save model checkpoints
    save_model(pan_cancer_model,
               save_dir / save_name)


    # plot loss curves
    plot_loss(training_losses, testing_losses,
              save_to=save_dir / loss_plot_name,
              show_plot=True)

    # evaluate model
    training_evaluate_metrics = evaluate_multiclass_classifier(model=pan_cancer_model,
                                                               data=training_data_tensor,
                                                               truth=training_labels_tensor,
                                                               classes=10)
    testing_evaluate_metrics = evaluate_multiclass_classifier(model=pan_cancer_model,
                                                              data=testing_data_tensor,
                                                              truth=testing_labels_tensor,
                                                              classes=10)

    final_train_acc = training_evaluate_metrics["top1_acc"]
    final_test_acc = testing_evaluate_metrics["top1_acc"]
    top3_acc = testing_evaluate_metrics["top3_acc"]
    top_5_acc = testing_evaluate_metrics["top5_acc"]

    final_train_area = training_evaluate_metrics["auroc"]
    final_test_area = testing_evaluate_metrics["auroc"]

    train_fpr, train_tpr, train_threshold = training_evaluate_metrics["roc"]
    test_fpr, test_tpr, test_threshold = testing_evaluate_metrics["roc"]

    # confusion
    conf_mat = testing_evaluate_metrics["confusion_matrix"]

    # recall on testing part
    rc = testing_evaluate_metrics["recall"]

    # precision on testing part
    prec = testing_evaluate_metrics["precision"]
    f1 = testing_evaluate_metrics["f_one"]


    # class specific metrics
    top1_acc_by_class = testing_evaluate_metrics["class_top1_acc"]
    top3_acc_by_class = testing_evaluate_metrics["class_top3_acc"]
    top_5_acc_by_class = testing_evaluate_metrics["class_top5_acc"]
    auroc_by_class = testing_evaluate_metrics["class_auroc"]
    recall_by_class = testing_evaluate_metrics["class_recall"]
    precision_by_class = testing_evaluate_metrics["class_precision"]
    f1_by_class = testing_evaluate_metrics["class_f_one"]


    # report metrics
    print("------------------------------------")
    print(f"Training Dataset")
    print(f"Accuracy: {training_evaluate_metrics['micro_top1_acc'] * 100 :.2f}({final_train_acc * 100:.2f})%")
    print(f"AUROC: {final_train_area:.5f}")
    print("------------------------------------")
    print(f"Testing Dataset")
    print(f"Accuracy: {testing_evaluate_metrics['micro_top1_acc'] * 100 :.2f}({final_test_acc * 100:.2f})%")
    print(f"Average AUROC: {final_test_area:.5f}")
    print(f"Recall: {rc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print("------------------------------------\n")

    # compile metric
    metrics = {"Model Name": pan_cancer_model.__class__.__name__,
               "Epochs": epochs,
               "auroc": final_test_area,
               "top1_acc": final_test_acc,
               "top3_acc": top3_acc,
               "top_5_acc": top_5_acc,
               "recall": rc,
               "precision": prec,
               "f_one": f1,
               "micro_top1_acc": testing_evaluate_metrics["micro_top1_acc"],
               "micro_top3_acc": testing_evaluate_metrics["micro_top3_acc"],
               "micro_top5_acc": testing_evaluate_metrics["micro_top5_acc"]}

    # add by class metrics
    for i in range(10):
        # for each class, report top1 top3 top5 accuracy, auroc, recall, precision, f1
        # testing values
        metrics[PAN_CANCER_DICT[i]] = {"top1_acc": top1_acc_by_class[i].item(),
                                       "top3_acc": top3_acc_by_class[i].item(),
                                       "top_5_acc": top_5_acc_by_class[i].item(),
                                       "auroc": auroc_by_class[i].item(),
                                       "recall": recall_by_class[i].item(),
                                       "precision": precision_by_class[i].item(),
                                       "f_one": f1_by_class[i].item()}

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
             save_to=save_dir / test_roc_name,
             auroc_by_class=auroc_by_class)

    plot_confusion_matrix(conf_mat,
                          save_to=save_dir / conf_mat_name,
                          indices=PAN_CANCER_LABELS)

    return pan_cancer_model


if __name__ == "__main__":
    model = PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)
    train_pan_cancer(pan_cancer_model=model,
                     epochs=40)
