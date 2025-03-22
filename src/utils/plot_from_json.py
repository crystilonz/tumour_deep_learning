from tarfile import TruncatedHeaderError

from data_manipulation.pancancer_from_csv import PAN_CANCER_LABELS
import json
import glob
from pathlib import Path
import torch
from utils.plotting import per_class_f1_plot, datasets_f1_plot
import os

def plot_per_class_f1_from_metrics(metrics, save_to=None, show_plot=True):
    f1 = []
    for c in PAN_CANCER_LABELS:
        f1.append(metrics[c]['f_one'])
    f1_tensor = torch.tensor(f1)
    per_class_f1_plot(f1_score=f1_tensor,
                      class_list=PAN_CANCER_LABELS,
                      save_to=save_to,
                      show_plot=show_plot)

def plot_per_class_f1_from_json(json_path, save_to=None, show_plot=True, save_plot=True):
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    if save_plot:
        if save_to is None:
            # resolve
            json_path = Path(json_path)
            save_to = json_path.parent / 'per_class_f1.png'
        plot_per_class_f1_from_metrics(metrics=metrics,
                                       save_to=save_to,
                                       show_plot=show_plot)
    else:
        plot_per_class_f1_from_metrics(metrics=metrics,
                                       save_to=None,
                                       show_plot=show_plot)


def plot_per_class_f1_from_dir(root_dir, file_name='metrics.json'):
    file_list = glob.glob('**/' + file_name,
                          root_dir = root_dir)
    for file in file_list:
        file_abs = os.path.join(root_dir, file)
        plot_per_class_f1_from_json(json_path=file_abs,
                                    save_to=None,
                                    show_plot=True,
                                    save_plot=True)

def plot_datasets_f1_from_metrics(cv_metrics, ext_metrics, gtex_metrics,
                                  class_list, save_to=None, show_plot=True):

    cv_f1s = []
    ext_f1s = []
    gtex_f1s = []

    for class_name in class_list:
        cv_f1s.append(cv_metrics[class_name]['f_one'])
        ext_f1s.append(ext_metrics[class_name]['f_one'])
        gtex_f1s.append(gtex_metrics[class_name]['f_one'])

    datasets_f1_plot(cv_f1s, ext_f1s, gtex_f1s, class_list, save_to, show_plot)

def plot_datasets_f1_from_json(cv_json, ext_json, gtex_json, class_list=None, save_to=None, show_plot=True, save_plot=True):
    with open(cv_json, 'r') as f:
        cv_metrics = json.load(f)
    with open(ext_json, 'r') as f:
        ext_metrics = json.load(f)
    with open(gtex_json, 'r') as f:
        gtex_metrics = json.load(f)

    if class_list is None:
        class_list = PAN_CANCER_LABELS

    if save_plot:
        if save_to is None:
            save_to = Path(Path(cv_json).parent / 'f1_datasets.png')
        plot_datasets_f1_from_metrics(cv_metrics=cv_metrics,
                                      ext_metrics=ext_metrics,
                                      gtex_metrics=gtex_metrics,
                                      class_list=class_list,
                                      save_to=save_to,
                                      show_plot=show_plot)
    else:
        plot_datasets_f1_from_metrics(cv_metrics=cv_metrics,
                                      ext_metrics=ext_metrics,
                                      gtex_metrics=gtex_metrics,
                                      class_list=class_list,
                                      save_to=None,
                                      show_plot=show_plot)

if __name__ == '__main__':
    plot_per_class_f1_from_dir(root_dir=Path(__file__).parent.parent / 'gtex_validation_results')
    plot_per_class_f1_from_dir(root_dir=Path(__file__).parent.parent / 'external_validation_results')

    plot_datasets_f1_from_json(cv_json=Path(__file__).parent.parent / 'validation_models' / 'leaky_pancancer' / 'best_metrics.json',
                               ext_json=Path(__file__).parent.parent / 'external_validation_results' / 'LeakyPanCancerClassifier' / 'metrics.json',
                               gtex_json=Path(__file__).parent.parent / 'gtex_validation_results' / 'LeakyPanCancerClassifier' / 'metrics.json',)

