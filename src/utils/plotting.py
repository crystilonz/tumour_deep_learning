import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
import seaborn as sn
import pandas as pd
from typing import Any, List
import os
import shap
from typing import Literal
from utils.shap_utils import shap_beeswarm_bar_pancancer, shap_waterfall_pancancer
import tqdm
import matplotlib.axes

ENV_SHOW_PLOT = os.environ.get('ENV_SHOW_PLOT')
if ENV_SHOW_PLOT:
    ENV_SHOW_PLOT = False if ENV_SHOW_PLOT == 'False' else True
else:
    ENV_SHOW_PLOT = True

# SHAP PLOT NAMES
SHAP_PLOT_NAMES = {"waterfall_correct": "shap_waterfall_correct_plot",
                   "waterfall_incorrect": "shap_waterfall_incorrect_plot",
                   "waterfall_all": "shap_waterfall_all_plot",
                   "beeswarm_bar_positive": "shap_beeswarm_bar_positive_plot",
                   "beeswarm_bar_negative": "shap_beeswarm_bar_negative_plot",
                   "beeswarm_bar_all": "shap_beeswarm_bar_all_plot"}


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
             auroc=None,
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
        title += f" (average AUC = {auroc:.05f})" if auroc else ""
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


def plot_imbalanced_confusion_matrix(cm: np.ndarray | torch.Tensor,
                                     save_to: str | Path = None,
                                     row_index: List[Any] = None,
                                     column_index: List[Any] = None,
                                     row_label: str = 'True label',
                                     column_label: str = 'Predicted label',
                                     row_to_delete: int | tuple[int] = None,
                                     column_to_delete: int | tuple[int] = None,
                                     plot_title: str = 'Confusion Matrix',
                                     show_plot: bool = True) -> None:
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()

    # delete rows
    if row_to_delete is not None:
        cm = np.delete(cm, row_to_delete, axis=0)

    # delete columns
    if column_to_delete is not None:
        cm = np.delete(cm, column_to_delete, axis=1)

    if row_index is None:
        row_index = range(cm.shape[0])

    if column_index is None:
        column_index = range(cm.shape[1])

    df_cm = pd.DataFrame(cm, index=row_index, columns=column_index)
    plt.figure(figsize=(8, 8))
    sn.heatmap(df_cm, annot=True)
    plt.title(plot_title)
    plt.xlabel(column_label)
    plt.ylabel(row_label)

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def plot_shap_beeswarm_bar(m: torch.nn.Module,
                           data: torch.Tensor,
                           labels: torch.Tensor = None,
                           e: shap.DeepExplainer = None,
                           model_pred: Literal["positive", "negative", "both"] = "both",
                           show_plot: bool = True,
                           save_to: str | Path = None) -> None:
    shap_beeswarm_bar_pancancer(m, data, labels,
                                e=e, model_pred=model_pred)

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def plot_shap_waterfall(m: torch.nn.Module,
                        data: torch.Tensor,
                        labels: torch.Tensor,
                        e: shap.DeepExplainer = None,
                        correct: Literal["true", "false", "both"] = "true",
                        sample_names=None,
                        slide_names=None,
                        show_plot: bool = True,
                        save_to: str | Path = None) -> None:
    shap_waterfall_pancancer(m, data, labels,
                             e=e,
                             correct=correct,
                             sample_names=sample_names,
                             slide_names=slide_names)

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def plot_shap_all(m: torch.nn.Module,
                  data: torch.Tensor,
                  labels: torch.Tensor,
                  e: shap.DeepExplainer = None,
                  sample_names=None,
                  slide_names=None,
                  show_plot: bool = True,
                  save_to_dir: str | Path = None,
                  plot_waterfall: bool = True,
                  use_tqdm=False) -> None:
    if e is None:
        e = shap.DeepExplainer(m, data)

    saving = False if save_to_dir is None else True

    if saving:
        if not save_to_dir.exists():
            save_to_dir.mkdir(parents=True)

    if use_tqdm:
        progress = tqdm.tqdm(total=6, desc="SHAP Progress")
    else:
        progress = None

    # waterfall plots
    if plot_waterfall:
        shap_waterfall_pancancer(m, data, labels,
                                 e=e,
                                 correct="true",
                                 sample_names=sample_names,
                                 slide_names=slide_names)
        if saving:
            plt.savefig(save_to_dir / SHAP_PLOT_NAMES["waterfall_correct"])

        if show_plot and ENV_SHOW_PLOT:
            plt.show()
        plt.close()
        if use_tqdm: progress.update(1)

        shap_waterfall_pancancer(m, data, labels,
                                 e=e,
                                 correct="false",
                                 sample_names=sample_names,
                                 slide_names=slide_names)

        if saving:
            plt.savefig(save_to_dir / SHAP_PLOT_NAMES["waterfall_incorrect"])

        if show_plot and ENV_SHOW_PLOT:
            plt.show()
        plt.close()
        if use_tqdm: progress.update(1)

        shap_waterfall_pancancer(m, data, labels,
                                 e=e,
                                 correct="both",
                                 sample_names=sample_names,
                                 slide_names=slide_names)
        if saving:
            plt.savefig(save_to_dir / SHAP_PLOT_NAMES["waterfall_all"])
        if show_plot and ENV_SHOW_PLOT:
            plt.show()
        plt.close()
        if use_tqdm: progress.update(1)
    else:
        if use_tqdm: progress.update(3)

    # beeswarm/bar
    shap_beeswarm_bar_pancancer(m, data, labels,
                                e=e,
                                model_pred="positive")
    if saving:
        plt.savefig(save_to_dir / SHAP_PLOT_NAMES["beeswarm_bar_positive"])
    if show_plot and ENV_SHOW_PLOT:
        plt.show()
    plt.close()
    if use_tqdm: progress.update(1)

    shap_beeswarm_bar_pancancer(m, data, labels,
                                e=e,
                                model_pred="negative")
    if saving:
        plt.savefig(save_to_dir / SHAP_PLOT_NAMES["beeswarm_bar_negative"])
    if show_plot and ENV_SHOW_PLOT:
        plt.show()
    plt.close()
    if use_tqdm: progress.update(1)

    shap_beeswarm_bar_pancancer(m, data, labels,
                                e=e,
                                model_pred="both")
    if saving:
        plt.savefig(save_to_dir / SHAP_PLOT_NAMES["beeswarm_bar_all"])
    if show_plot and ENV_SHOW_PLOT:
        plt.show()
    plt.close()
    if use_tqdm:
        progress.update(1)
        progress.close()


def k_folds_loss_to_pandas(loss_folds: [[float]]) -> pd.DataFrame:
    df = None
    for fold_num, fold in enumerate(loss_folds):
        for epoch, loss in enumerate(fold):
            this_epoch = pd.DataFrame({"fold": fold_num, "epoch": epoch, "loss": loss},
                                      columns=["fold", "epoch", "loss"],
                                      index=[0])
            df = pd.concat([df, this_epoch], ignore_index=True)

    return df


def plot_loss_k_folds(training_loss_folds: [[float]],
                      testing_loss_folds: [[float]],
                      save_to: str | Path = None,
                      show_plot: bool = True) -> None:
    # testing dataframe
    training_df = k_folds_loss_to_pandas(training_loss_folds)
    training_df['phase'] = "training"

    testing_df = k_folds_loss_to_pandas(testing_loss_folds)
    testing_df['phase'] = "testing"

    combined_df = pd.concat([training_df, testing_df])

    plt.figure(figsize=(7, 7))

    # training
    sn.lineplot(combined_df, x="epoch", y="loss", hue="phase", errorbar="sd", err_style="band")
    plt.ylim(0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    title = f"{len(training_loss_folds)}-Fold Validation Loss"
    plt.title(title)

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def per_class_acc_barplot(acc1: torch.Tensor,
                          acc3: torch.Tensor,
                          acc5: torch.Tensor,
                          class_list: list[str],
                          save_to: str | Path = None,
                          show_plot: bool = True) -> None:
    plt.figure(figsize=(14, 7))

    acc1 = acc1.numpy()
    acc3 = acc3.numpy()
    acc5 = acc5.numpy()

    classes = []
    acc_top = []
    vals = []

    for num, cls in enumerate(class_list):
        # append top accs
        classes.append(cls)
        acc_top.append(1)
        vals.append(acc1[num])

        classes.append(cls)
        acc_top.append(3)
        vals.append(acc3[num])

        classes.append(cls)
        acc_top.append(5)
        vals.append(acc5[num])

    plt_dict = {'Class': classes,
                'Top-k Acc': acc_top,
                'Accuracy': vals}

    plt_df = pd.DataFrame(plt_dict)

    sn.barplot(data=plt_df, x="Class", y="Accuracy", hue="Top-k Acc")
    plt.title('Per-class Accuracy')
    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def per_class_auroc_plot(auroc: torch.Tensor,
                         class_list: list[str],
                         save_to: str | Path = None,
                         show_plot: bool = True) -> None:
    plt.figure(figsize=(14, 7))
    auroc = auroc.numpy()

    ax = sn.barplot(x=class_list, y=auroc)
    ax.bar_label(ax.containers[0], fontsize=16, fmt='%.3f')

    plt.ylim(top=1.05)
    plt.title('Per-class AUROC')

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def per_class_f1_plot(f1_score: torch.Tensor,
                      class_list: list[str],
                      save_to: str | Path = None,
                      show_plot: bool = True) -> None:
    plt.figure(figsize=(14, 7))
    f1_score = f1_score.cpu().numpy()

    ax = sn.barplot(x=class_list, y=f1_score, color='g')
    ax.bar_label(ax.containers[0], fontsize=16, fmt='%.3f')

    plt.ylim(top=1.05)
    plt.title('Per-class F1 Score')

    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def datasets_f1_plot(f1_score_cv,
                     f1_score_ext,
                     f1_score_gtex,
                     class_list: list[str],
                     save_to: str | Path = None,
                     show_plot: bool = True) -> None:
    plt.figure(figsize=(14, 7))

    classes = []
    dataset = []
    f1_scores = []

    for num, cls in enumerate(class_list):
        classes.append(cls)
        dataset.append("TCGA")
        f1_scores.append(f1_score_cv[num])

        classes.append(cls)
        dataset.append("External")
        f1_scores.append(f1_score_ext[num])

        classes.append(cls)
        dataset.append("GTEx")
        f1_scores.append(f1_score_gtex[num])

    plt_dict = {'Class': classes,
                'F1 Score': f1_scores,
                'Dataset': dataset}

    plt_df = pd.DataFrame(plt_dict)

    sn.barplot(data=plt_df, x="Class", y="F1 Score", hue="Dataset", palette='viridis')
    plt.title('Per-class F1 Score Between Datasets')
    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()


def datasets_aurocs_plot(aurocs_cv,
                         aurocs_ext,
                         aurocs_gtex,
                         class_list: list[str],
                         save_to: str | Path = None,
                         show_plot: bool = True) -> None:
    plt.figure(figsize=(14, 7))

    classes = []
    dataset = []
    aurocs = []

    for num, cls in enumerate(class_list):
        classes.append(cls)
        dataset.append("TCGA")
        aurocs.append(aurocs_cv[num])

        classes.append(cls)
        dataset.append("External")
        aurocs.append(aurocs_ext[num])

        classes.append(cls)
        dataset.append("GTEx")
        aurocs.append(aurocs_gtex[num])

    plt_dict = {'Class': classes,
                'AUROC': aurocs,
                'Dataset': dataset}

    plt_df = pd.DataFrame(plt_dict)

    sn.barplot(data=plt_df, x="Class", y="AUROC", hue="Dataset", palette='plasma')
    plt.title('Per-class AUROCs Between Datasets')
    if save_to is not None:
        plt.savefig(save_to)

    if show_plot and ENV_SHOW_PLOT:
        plt.show()

    plt.close()
