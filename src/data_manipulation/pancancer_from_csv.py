import csv
import os
import numpy as np


def get_data_from_csv(dir_path: str):
    file_list = os.listdir(dir_path)
    sample_list, slide_list, WSI_tensor, WSI_label = None, None, None, None


    for file in file_list:
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'r') as f:
            np_csv = np.genfromtxt(f, delimiter=',', dtype=np.dtypes.StringDType)
            np_csv = np_csv[1:]  # remove header
        sample_list = np_csv[:, 0] if not sample_list else np.concatenate((sample_list, np_csv[:, 1]), axis=0)
        slide_list = np_csv[:, 1] if not slide_list else np.concatenate((slide_list, np_csv[:, 1]), axis=0)
        WSI_tensor = np_csv[:, 2: -2].astype(np.float32) if not WSI_tensor else np.concatenate((WSI_tensor, np.float32(np_csv[:, 2: -2])), axis=0)
        WSI_label = np_csv[:, -1].astype(np.int32) if not WSI_label else np.concatenate((WSI_label, np.int32(np_csv[:, -1])), axis=0)

    return sample_list, slide_list, WSI_tensor, WSI_label


if __name__ == '__main__':
    DATA_DIR = "src/datasets/pancancer_WSI_representation"
    samples, slides, tensor, labels = get_data_from_csv(DATA_DIR)
    print(tensor.shape)
