import csv
import os
import numpy as np

PAN_CANCER_DICT = {0: "BLCA",
                   1: "BRCA",
                   2: "CESC",
                   3: "COAD",
                   4: "LUAD",
                   5: "LUSC",
                   6: "PRAD",
                   7: "SKCM",
                   8: "STAD",
                   9: "UCEC"}

PAN_CANCER_LABELS: list[str] = [PAN_CANCER_DICT[i] for i in range(0, 10)]


def get_pancancer_data_from_csv(dir_path: str):
    file_list = os.listdir(dir_path)
    sample_list, slide_list, WSI_tensor, WSI_label = None, None, None, None


    for file in file_list:
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'r') as f:
            np_csv = np.genfromtxt(f, delimiter=',', dtype=np.dtypes.StringDType)
            np_csv = np_csv[1:]  # remove header
        sample_list = np_csv[:, 0] if not sample_list else np.concatenate((sample_list, np_csv[:, 1]), axis=0)
        slide_list = np_csv[:, 1] if not slide_list else np.concatenate((slide_list, np_csv[:, 1]), axis=0)
        WSI_tensor = np_csv[:, 2: -1].astype(np.float32) if not WSI_tensor else np.concatenate((WSI_tensor, np.float32(np_csv[:, 2: -2])), axis=0)
        WSI_label = np_csv[:, -1].astype(np.int32) if not WSI_label else np.concatenate((WSI_label, np.int32(np_csv[:, -1])), axis=0)

    return sample_list, slide_list, WSI_tensor, WSI_label


if __name__ == '__main__':
    DATA_DIR = "../datasets/pancancer_WSI_representation"
    samples, slides, tensor, labels = get_pancancer_data_from_csv(DATA_DIR)
    for i in range(10):
        print(f"Number of samples in class {PAN_CANCER_DICT[i]}: {np.sum(labels == i)}")


