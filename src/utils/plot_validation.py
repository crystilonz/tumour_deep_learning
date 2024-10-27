import json
import numpy as np
import seaborn as sn
import pandas as pd
from pathlib import Path

def plot_validation(parent_dir: Path,
                    metrics_file_name: Path | str,
                    plot_save_dir: Path):

    model_names: list[str] = []  # list of model names
    losses: list[list[float]] = []  # list of list of loss from each fold
    top1_accs: list[list[float]] = []  # list of list of accs
    aurocs: list[list[float]] = []  #


    # scour the directory for metrics file name
    sub_directories = [f for f in parent_dir.iterdir() if f.is_dir()]
    for sub_dir in sub_directories:
        if not (sub_dir / metrics_file_name).exists():
            # if does not contain metric then skip
            continue

        with sub_dir.joinpath(metrics_file_name).open() as metrics_file:
            metrics_data = metrics_file.read()
            metrics_dict = json.loads(metrics_data)



    pass
    #TODO
