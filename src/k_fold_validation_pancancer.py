from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from pathlib import Path
import numpy as np

from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_DICT, PAN_CANCER_LABELS
from models.pan_cancer_classifier import PanCancerClassifier
from utils.datadump import save_to_json
from utils.multiclass_evaluate import accuracy, auroc, roc, confusion_matrix, recall, precision, f_beta, \
    evaluate_multiclass_classifier
from utils.plotting import plot_loss, plot_roc, plot_confusion_matrix
from utils.training import train_model, save_model
from typing import Literal

DEFAULT_DATA_DIR = Path(__file__).parent / "datasets" / "pancancer_WSI_representation"
DEFAULT_VALIDATION_RESULTS_PARENT = Path(__file__).parent / "validation_models"
DEFAULT_VALIDATION_MODEL_NAME = "pancancer_classifier"
DEFAULT_BEST_MODEL_NAME = "pancancer_classifier_checkpoints"
DEFAULT_VALIDATION_RESULTS_NAME = "k_fold_validation.json"
DEFAULT_BEST_MODEL_METRICS_NAME = "best_metrics.json"
DEFAULT_PLOT_NAME = "best_loss_plot"
DEFAULT_ROC_NAME = "best_roc"
DEFAULT_CONFUSION_MATRIX_NAME = "best_confusion_matrix"
DEFAULT_EPOCH = 30
DEFAULT_HIDDEN_SIZE = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_FOLD = 5


def k_fold_validation_pancancer(
        model_creating_lambda,
        mode: Literal["normal", "stratified"] = "stratified",
        datadir: str | Path = DEFAULT_DATA_DIR,
        epochs: int = DEFAULT_EPOCH,
        k: int = DEFAULT_FOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        results_parent: str | Path = DEFAULT_VALIDATION_RESULTS_PARENT,
        results_dir: str | Path = DEFAULT_VALIDATION_MODEL_NAME,
        best_model_save_name: str | Path = DEFAULT_BEST_MODEL_NAME,
        best_model_metrics_save_name: str | Path = DEFAULT_BEST_MODEL_METRICS_NAME,
        validation_results_save_name: str | Path = DEFAULT_VALIDATION_RESULTS_NAME,
        best_loss_plot_save_name: str | Path = DEFAULT_PLOT_NAME,
        best_roc_plot_save_name: str | Path = DEFAULT_ROC_NAME,
        best_conf_mat_save_name: str | Path = DEFAULT_CONFUSION_MATRIX_NAME
) -> torch.nn.Module:
    """
    :param model_creating_lambda: lambda function for creating model. calling x must return a new instance of the model
    :param mode: "normal" for k-fold, and "stratified" for stratified k-fold. Use stratified for imbalanced class
    :param datadir: Data directory
    :param epochs:  Number of epochs
    :param k: K in k-fold validation. Default is 20
    :param batch_size: Batch size for training
    :param results_parent: parent directory of the validation results (of all models)
    :param results_dir: directory which will contain the validation output
    :param best_model_save_name: name of the checkpoint file which will contain the checkpoint of the best performing model (smallest testing loss)
    :param best_model_metrics_save_name: name of the metrics file for the best performing model (smallest testing loss)
    :param validation_results_save_name: file name for the results of validation
    :param best_loss_plot_save_name: file name for the loss plot
    :param best_roc_plot_save_name: file name for the roc plot
    :param best_conf_mat_save_name: file name for the confusion matrix plot
    :return: the best model from k-fold validation
    """

    samples, slides, data, labels = get_pancancer_data_from_csv(datadir)
    # turn to tensors
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)

    # find the weights for the loss function
    total_label = len(labels)
    weights = [total_label / (np.sum(labels == i) * 10) for i in range(10)]  # use this in loss function

    best_model = None  # to remember the best model instance
    best_loss = None  # remember the best testing loss
    from_fold = None
    best_metrics = None  # remember metrics

    # keep track of the fold
    training_losses_folds = []
    testing_losses_folds = []
    loss_list = []
    acc1_list = []
    acc3_list = []
    acc5_list = []
    recall_list = []
    precision_list = []
    auroc_list = []
    f_one_list = []

    micro_acc1_list = []
    micro_acc3_list = []
    micro_acc5_list = []

    if mode == "normal":
        kf = KFold(n_splits=k, shuffle=True)
    elif mode == "stratified":
        kf = StratifiedKFold(n_splits=k, shuffle=True)
    else:
        # INVALID
        raise(ValueError("mode must be either 'normal' or 'stratified'"))

    print("\n" + "=" * 100)
    for fold, (train_index, test_index) in enumerate(kf.split(data, labels)):
        print(f"Fold #{fold + 1}")

        # instantiate model
        model = model_creating_lambda()

        # put in tensor dataset
        dataset = TensorDataset(data_tensor, labels_tensor)

        # define data loader
        train_dataloader = DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_index))

        test_dataloader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_index))

        # loss function
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))

        # optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        training_losses, testing_losses = train_model(model, optimizer, loss_fn, train_dataloader, test_dataloader,
                                                      epochs)
        final_training_loss = training_losses[-1]
        final_testing_loss = testing_losses[-1]

        # evaluate metrics
        metrics = evaluate_multiclass_classifier(model=model,
                                                 data=data_tensor[test_index],
                                                 truth=labels_tensor[test_index],
                                                 classes=10)

        if best_loss is None or final_training_loss < best_loss:
            # this is the new best model
            best_loss = final_training_loss
            best_model = model
            from_fold = fold
            best_metrics = metrics

        # append the list of losses
        training_losses_folds.append(training_losses)
        testing_losses_folds.append(testing_losses)

        # loss append
        loss_list.append(final_testing_loss)

        # from metrics
        acc1_list.append(metrics['top1_acc'])
        acc3_list.append(metrics['top3_acc'])
        acc5_list.append(metrics['top5_acc'])
        recall_list.append(metrics['recall'])
        precision_list.append(metrics['precision'])
        auroc_list.append(metrics['auroc'])
        f_one_list.append(metrics['f_one'])

        # micro
        micro_acc1_list.append(metrics['micro_top1_acc'])
        micro_acc3_list.append(metrics['micro_top3_acc'])
        micro_acc5_list.append(metrics['micro_top5_acc'])

        # report
        print(f"Final Loss: {final_testing_loss:.5f}")
        print(f"Top1 Accuracy: {metrics['micro_top1_acc'] * 100:.02f}({metrics['top1_acc'] * 100:.02f}) %")
        print(f"Top3 Accuracy: {metrics['micro_top3_acc'] * 100:.02f}({metrics['top3_acc'] * 100:.02f}) %")
        print(f"AUROC: {metrics['auroc']:.5f}")
        print("-" * 80)

    # after all folds --> get avg
    avg_loss = np.mean(loss_list)
    avg_acc1 = np.mean(acc1_list)
    avg_acc3 = np.mean(acc3_list)
    avg_acc5 = np.mean(acc5_list)
    avg_recall = np.mean(recall_list)
    avg_precision = np.mean(precision_list)
    avg_auroc = np.mean(auroc_list)
    avg_f_one = np.mean(f_one_list)

    # micro
    avg_micro_acc1 = np.mean(micro_acc1_list)
    avg_micro_acc3 = np.mean(micro_acc3_list)
    avg_micro_acc5 = np.mean(micro_acc5_list)

    # compile metrics
    validation_metrics = {"Model Name": best_model.__class__.__name__,
                          "Folds": k,
                          "Epochs": epochs,
                          "avg_loss": avg_loss,
                          "avg_acc1": avg_acc1,
                          "avg_acc3": avg_acc3,
                          "avg_acc5": avg_acc5,
                          "avg_recall": avg_recall,
                          "avg_precision": avg_precision,
                          "avg_auroc": avg_auroc,
                          "avg_f_one": avg_f_one,

                          # micro data
                          "avg_micro_acc1": avg_micro_acc1,
                          "avg_micro_acc3": avg_micro_acc3,
                          "avg_micro_acc5": avg_micro_acc5,


                          # data for each fold
                          "folds_losses": loss_list,
                          "folds_acc1": acc1_list,
                          "folds_acc3": acc3_list,
                          "folds_acc5": acc5_list,
                          "folds_recall": recall_list,
                          "folds_precision": precision_list,
                          "folds_auroc": auroc_list,
                          "folds_f_one": f_one_list,

                          # micro
                          "folds_micro_acc1": micro_acc1_list,
                          "folds_micro_acc3": micro_acc3_list,
                          "folds_micro_acc5": micro_acc5_list,
                          }

    # metrics for the best model
    best_saved_metrics = {"Model Name": best_model.__class__.__name__,
                          "Folds": k,
                          "From Fold": from_fold,
                          "Epochs": epochs,
                          "auroc": best_metrics['auroc'],
                          "top1_acc": best_metrics['top1_acc'],
                          "top3_acc": best_metrics['top3_acc'],
                          "top_5_acc": best_metrics['top5_acc'],
                          "recall": best_metrics['recall'],
                          "precision": best_metrics['precision'],
                          "f_one": best_metrics['f_one'],

                          "micro_top1_acc": best_metrics['micro_top1_acc'],
                          "micro_top3_acc": best_metrics['micro_top3_acc'],
                          "micro_top5_acc": best_metrics['micro_top5_acc'],
                          }

    # class specific metrics
    for i in range(10):
        # for each class, report top1 top3 top5 accuracy, auroc, recall, precision, f1
        # testing values
        best_saved_metrics[PAN_CANCER_DICT[i]] = {"top1_acc": best_metrics['class_top1_acc'][i].item(),
                                                  "top3_acc": best_metrics['class_top3_acc'][i].item(),
                                                  "top_5_acc": best_metrics['class_top5_acc'][i].item(),
                                                  "auroc": best_metrics['class_auroc'][i].item(),
                                                  "recall": best_metrics['class_recall'][i].item(),
                                                  "precision": best_metrics['class_precision'][i].item(),
                                                  "f_one": best_metrics['class_f_one'][i].item()
                                                  }

    # set save directory
    save_dir = results_parent / results_dir

    # save best checkpoint
    save_model(best_model, save_dir / best_model_save_name)

    # save metrics
    save_to_json(validation_metrics, save_dir / validation_results_save_name)
    save_to_json(best_saved_metrics, save_dir / best_model_metrics_save_name)

    # plot loss, roc, conf mat
    plot_loss(training_losses_folds[from_fold],
              testing_losses_folds[from_fold],
              save_to=save_dir / best_loss_plot_save_name)

    fpr, tpr, threshold = best_metrics['roc']
    plot_roc(false_pos=fpr,
             true_pos=tpr,
             threshold=threshold,
             auroc_by_class=best_metrics['class_auroc'],
             auroc=best_metrics['auroc'],
             title="ROC of best model",
             save_to=save_dir / best_roc_plot_save_name)

    plot_confusion_matrix(best_metrics['confusion_matrix'],
                          save_to=save_dir / best_conf_mat_save_name,
                          indices=PAN_CANCER_LABELS)

    # report
    print("-" * 80)
    print(f"Performed {k}-fold cross validation on {best_model.__class__.__name__}, with {epochs} epochs:")
    print(f"\tAverage Loss: {avg_loss:.5f}")
    print(f"\tAverage Acc1: {avg_micro_acc1 * 100:.02f}({avg_acc1 * 100:.02f}) %")
    print(f"\tAverage Acc3: {avg_micro_acc3 * 100:.02f}({avg_acc3 * 100:.02f}) %")
    print(f"\tAverage Acc5: {avg_micro_acc5 * 100:.02f}({avg_acc5 * 100:.02f}) %")
    print(f"\tAverage Recall: {avg_recall:.5f}")
    print(f"\tAverage Precision:{avg_precision:.5f}")
    print(f"\tAverage AUROC: {avg_auroc:.5f}")
    print(f"\tAverage F1: {avg_f_one:.5f}")
    print("-" * 80)
    print("=" * 100 + "\n")


    return best_model

if __name__ == '__main__':
    # model creation lambda
    instantiate_pancancer = lambda : PanCancerClassifier(input_size=34,
                                                         hidden_size=42,
                                                         output_size=10)
    # call
    k_fold_validation_pancancer(instantiate_pancancer)