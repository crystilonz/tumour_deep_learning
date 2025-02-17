import h5py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from data_manipulation.pancancer_from_csv import PAN_CANCER_DICT
from typing import Literal
from pathlib import Path
import matplotlib.pyplot as plt


def pancancer_inference(model: nn.Module,
                        data: torch.Tensor,
                        device: Literal['cpu', 'cuda'] | None = None) -> str:
    if device is None:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    data.to(device)
    model.eval()
    with torch.no_grad():
        prob_preds = model(data)
        class_prediction = torch.argmax(prob_preds).item()

    return PAN_CANCER_DICT[class_prediction]


def process_hdf5_to_full_names(hdf5_path: str,
                               hdf5_name_field: str | list[str] | tuple[str],
                               hdf5_name_concat: str) -> pd.DataFrame:
    with h5py.File(hdf5_path, 'r') as f:
        if isinstance(hdf5_name_field, list) or isinstance(hdf5_name_field, tuple):
            columns = [pd.DataFrame(np.array(f[col_name]), dtype='string') for col_name in hdf5_name_field]
            concatenated = pd.concat(columns, axis=1)
            hdf5_complete_name = concatenated.apply(lambda x: hdf5_name_concat.join(x), axis=1)
        else:
            hdf5_complete_name = pd.Series(np.array(f[hdf5_name_field]), dtype='string')

    hdf5_length = len(hdf5_complete_name)
    original_indices = pd.DataFrame(np.arange(hdf5_length), columns=['original_indices'])
    renamed_complete_name_pd = pd.DataFrame(hdf5_complete_name, columns=['full_names'])
    hdf5_complete_name = pd.concat([original_indices, renamed_complete_name_pd], axis=1)

    return hdf5_complete_name


def get_hdf5_index_from_name(full_name: str,
                             full_names_pd:pd.DataFrame):
    index = full_names_pd[full_names_pd['full_names'] == full_name]['original_indices']
    if len(index) == 0:
        print("No match in indices for", full_name)
        return None
    elif len(index) > 1:
        print("Multiple match in indices for", full_name)

    return int(index.item())

def get_image_by_index(index:int,
                       image_hdf5_path: str|Path,
                       image_field_name: str):
    with h5py.File(image_hdf5_path, 'r') as f:
        img = np.array(f[image_field_name][index])
    return img

def get_image_by_name(full_name: str,
                      full_names_pd:pd.DataFrame,
                      image_hdf5_path: str|Path,
                      image_field_name: str):
    index = get_hdf5_index_from_name(full_name, full_names_pd)
    if index is None:
        return None
    img = get_image_by_index(index, image_hdf5_path, image_field_name)
    return img

def show_image(img_array:np.ndarray,
               swap_RGB = False):
    if swap_RGB:
        img_array = img_array[:, :, [2, 1, 0]]
    plt.imshow(img_array)
    plt.axis('off')
    plt.show()

def process_csv_to_full_names(csv_path: str,
                              csv_name_field: str | list[str] | tuple[str],
                              csv_name_concat: str,
                              csv_leiden_field:str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if isinstance(csv_name_field, list) or isinstance(csv_name_field, tuple):
        columns = [pd.DataFrame(df[col_name]) for col_name in csv_name_field]
        concatenated = pd.concat(columns, axis=1)
        csv_complete_name = concatenated.apply(lambda x: csv_name_concat.join(x), axis=1)
    else:
        csv_complete_name = df[csv_name_field]

    leiden = pd.DataFrame(df[csv_leiden_field], columns=['leiden_2.0'])
    renamed_complete_name_pd = pd.DataFrame(csv_complete_name, columns=['full_names'])
    csv_complete_name = pd.concat([leiden, renamed_complete_name_pd], axis=1)
    return csv_complete_name
