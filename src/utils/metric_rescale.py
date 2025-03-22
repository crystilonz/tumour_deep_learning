import glob
import json
from pathlib import Path
from utils.datadump import save_to_json
import os

def rescale_metrics(metrics, class_num, class_actual):
    factor = float(class_num) / float(class_actual)
    return {
        "Model Name": metrics["Model Name"],
        "auroc": metrics["auroc"] * factor,
        "recall": metrics["recall"] * factor,
        "precision": metrics["precision"] * factor,
        "f_one": metrics["f_one"] * factor,
    }

def rescale_json(json_path, class_num, class_actual, out_path=None):
    if out_path is None:
        json_path = Path(json_path)
        out_path = json_path.parent / 'rescaled_metrics.json'
    with open(json_path, "r") as f:
        metrics = json.load(f)

    rescaled_metrics = rescale_metrics(metrics, class_num, class_actual)
    save_to_json(rescaled_metrics, out_path)

def rescale_dir(root_dir, class_num, class_actual, file_name='metrics.json'):
    file_list= glob.glob(pathname="**/"+file_name,
                         root_dir=root_dir)
    for file in file_list:
        file_abs = os.path.join(root_dir, file)
        rescale_json(file_abs, class_num, class_actual)

if __name__ == '__main__':
    rescale_dir(root_dir=Path(__file__).parent.parent / 'gtex_validation_results', class_num=10, class_actual=9)
    rescale_dir(root_dir=Path(__file__).parent.parent / 'external_validation_results', class_num=10, class_actual=7)