import torch
import torch.nn as nn
from pathlib import Path
from utils.multiclass_evaluate import evaluate_multiclass_classifier
from utils.datadump import save_to_json
from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_LABELS, PAN_CANCER_DICT
from utils.plotting import plot_roc, plot_confusion_matrix, per_class_auroc_plot, per_class_acc_barplot
import numpy as np

DEFAULT_DIRECTORY = Path(__file__).parent.parent / 'external_validation_results'
DEFAULT_EXT_DIRECTORY = Path(__file__).parent.parent / 'datasets' / 'external_pancancer'
METRIC_FILE_NAME = r'metrics.json'
CONFUSION_MATRIX_FILE_NAME = r'confusion_matrix'
ROC_FILE_NAME = r'roc'
CLASS_AUROC_FILE_NAME = r'class_auroc'
CLASS_ACC_FILE_NAME = r'class_accuracy'
CHECKPOINT_NAME = r'checkpoint.pt'


def validate_with_external_set(model: nn.Module,
                               checkpoint: Path | str,
                               external_csv: Path | str = DEFAULT_EXT_DIRECTORY,
                               output_dir: Path | str = None,
                               num_classes: int = None) -> dict:
    # where to save the results
    if output_dir is None:
        output_dir = DEFAULT_DIRECTORY / model.__class__.__name__
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir) / model.__class__.__name__
    else:
        output_dir = output_dir / model.__class__.__name__

    samples, slides, data, labels = get_pancancer_data_from_csv(external_csv)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))

    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)

    results = evaluate_multiclass_classifier(model, data_tensor, labels_tensor, classes=10)

    json_results = {"Model Name": model.__class__.__name__,
                    "auroc": results['auroc'],
                    "top1_acc": results['top1_acc'],
                    "top3_acc": results['top3_acc'],
                    "top5_acc": results['top5_acc'],
                    "recall": results['recall'],
                    "precision": results['precision'],
                    "f_one": results['f_one'],
                    "micro_top1_acc": results["micro_top1_acc"],
                    "micro_top3_acc": results["micro_top3_acc"],
                    "micro_top5_acc": results["micro_top5_acc"]}

    # add by class metrics
    for i in range(10):
        # for each class, report top1 top3 top5 accuracy, auroc, recall, precision, f1
        # testing values
        json_results[PAN_CANCER_DICT[i]] = {"top1_acc": results['class_top1_acc'][i].item(),
                                       "top3_acc": results['class_top3_acc'][i].item(),
                                       "top5_acc": results['class_top5_acc'][i].item(),
                                       "auroc": results['class_auroc'][i].item(),
                                       "recall": results['class_recall'][i].item(),
                                       "precision": results['class_precision'][i].item(),
                                       "f_one": results['class_f_one'][i].item()}

    # output
    save_to_json(json_results, output_dir / METRIC_FILE_NAME)

    fpr, tpr, thr = results['roc']
    if num_classes is None:
        auroc_class = results['auroc']
    else:
        auroc_class = np.sum(results['class_auroc'].cpu().numpy()) / num_classes
    plot_roc(fpr, tpr, thr,
             auroc_by_class=results['class_auroc'],
             auroc=auroc_class,
             save_to=output_dir / ROC_FILE_NAME,
             label_dict=PAN_CANCER_DICT)

    cm = results['confusion_matrix']
    plot_confusion_matrix(cm,
                          save_to=output_dir / CONFUSION_MATRIX_FILE_NAME,
                          indices=PAN_CANCER_LABELS)

    per_class_auroc_plot(results['class_auroc'], class_list=PAN_CANCER_LABELS,
                         save_to=output_dir / CLASS_AUROC_FILE_NAME)

    per_class_acc_barplot(acc1=results['class_top1_acc'],
                          acc3=results['class_top3_acc'],
                          acc5=results['class_top5_acc'],
                          class_list=PAN_CANCER_LABELS,
                          save_to=output_dir / CLASS_ACC_FILE_NAME)

    return results
