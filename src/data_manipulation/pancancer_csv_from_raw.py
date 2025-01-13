import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Literal


def count_leiden(df: pd.DataFrame,
                 combined_name: str,
                 out_size: int = 34,
                 slide_column_name: str = 'combined_slides',
                 leiden_column_name: str = 'leiden_2.0',
                 normalize: Literal['none', 'clr'] = 'clr') -> np.ndarray:
    """
    Count number of leiden cluster occurrence in the specified slide given the dataframe.
    :param df: DataFrame to extract leiden cluster occurrence from
    :param combined_name: the name of the combined slide to count the occurrence
    :param out_size: number of clusters expected. This will be the size of the output 1-D np.ndarray
    :param leiden_column_name: the name of the column in the dataframe that contains the leiden clusters
    :param slide_column_name: the name of the column in the dataframe that contains the slide name
    :param normalize: normalisation method. default is CLR
    :return: frequency counts. index 0 refers to count of type 0, and so on
    """
    df_slide = df[df[slide_column_name] == combined_name]
    freqs = df_slide[leiden_column_name].value_counts()
    freqs_ind = np.arange(out_size)
    freqs_np = pd.Series(data=freqs, index=freqs_ind).fillna(0).to_numpy()

    if normalize == 'none':
        return freqs_np
    elif normalize == 'clr':
        return clr_frequency(freqs_np)
    else:
        print('Unknown normalization method, defaulting to CLR')
        return clr_frequency(freqs_np)


def clr_frequency(freq_np: np.ndarray,
                  imputation: float = 1) -> np.ndarray:
    # change to faction of occurrence
    total_tiles = freq_np.sum()
    fraction_np = freq_np / total_tiles

    # zero imputation
    # fraction_np = fraction_np + imputation
    freq_np = freq_np + imputation

    # centered log ratio (CLR)
    # geo_mean = np.prod(fraction_np) ** (1.0 / freq_np.size)
    geo_mean = np.prod(freq_np) ** (1.0 / freq_np.shape[0])

    # return np.log(fraction_np / geo_mean)
    return np.log(freq_np / geo_mean)



def df_from_space_seperated_values(ssv_path: Path | str) -> pd.DataFrame:
    with open(ssv_path, 'r') as f:
        df = pd.read_csv(f, sep=' ', header=None)
    df.rename(columns={df.columns[0]: 'combined_name', df.columns[1]: 'classification', df.columns[2]: 'label'},
              inplace=True)
    names_df = df['combined_name'].apply((lambda x: pd.Series(get_sample_slide_name_from_combined(x))))
    names_df.rename(columns={names_df.columns[0]: 'sample_name', names_df.columns[1]: 'slide_name'}, inplace=True)
    ddf = names_df.join(df)
    return ddf


def get_sample_slide_name_from_combined(combined_str: str) -> [str]:
    parts = combined_str.split('_')
    sample = parts[0] + '_' + parts[1]
    return [sample, parts[2]]


def build_representation_vector(leiden_df: pd.DataFrame,
                                classes_df: pd.DataFrame,
                                leiden_slide_column_name: str = 'combined_slides',
                                leiden_leiden_column_name: str = 'leiden_2.0',
                                classes_slide_column_name: str = 'slide_name',
                                classes_sample_column_name: str = 'sample_name',
                                classes_combined_column_name: str = 'combined_slides',
                                classes_label_column_name: str = 'label') -> pd.DataFrame:
    """
    Build dataframe for csv. The columns are:
    samples,slides,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,labels
    :param leiden_df: DataFrame containing leiden representation of each tile
    :param classes_df: DataFrame containing list of slides and its truth (class label)
    :param leiden_slide_column_name: Column of leiden_df containing the slide name
    :param leiden_leiden_column_name: Column of leiden_df containing the leiden designation
    :param classes_slide_column_name: Column of classes_df containing the slide name
    :param classes_sample_column_name: Column of classes_df containing the sample name
    :param classes_combined_column_name: Column of classes_df containing the combined sample_slide name
    :param classes_label_column_name: Column of classes_df containing the label
    :return: DataFrame formatted for input to pancancer
    """
    samples_slides_df = pd.DataFrame({'samples': classes_df[classes_sample_column_name],
                                      'slides': classes_df[classes_slide_column_name]})

    count_df = classes_df[classes_combined_column_name].apply(lambda x: pd.Series(count_leiden(leiden_df, x,
                                                                                               out_size=34,
                                                                                               slide_column_name=leiden_slide_column_name,
                                                                                               leiden_column_name=leiden_leiden_column_name,
                                                                                               normalize='clr',
                                                                                               )
                                                                                  ))
    labels_df = pd.DataFrame({'labels': classes_df[classes_label_column_name],})

    return samples_slides_df.join(count_df).join(labels_df)


def build_csv_from_raw(leiden_csv: Path | str,
                       classes_csv: Path | str,
                       save_to: Path | str,
                       leiden_slide_column_name: str = 'combined_slides',
                       leiden_leiden_column_name: str = 'leiden_2.0',
                       classes_slide_column_name: str = 'slide_name',
                       classes_sample_column_name: str = 'sample_name',
                       classes_combined_column_name: str = 'combined_name',
                       classes_label_column_name: str = 'label',
                       is_classes_space_separated: bool = True):

    with open(leiden_csv, 'r') as leiden_f:
        leiden_df = pd.read_csv(leiden_f)

    with open(classes_csv, 'r') as classes_f:
        if is_classes_space_separated:
            classes_df = df_from_space_seperated_values(classes_csv)
        else:
            classes_df = pd.read_csv(classes_f)

    classes_df = filter_empty_tiles(leiden_df, classes_df,
                                    leiden_slide_column_name=leiden_slide_column_name,
                                    classes_combined_column_name=classes_combined_column_name,)


    if not isinstance(save_to, Path):
        save_to = Path(save_to)

    Path.mkdir(save_to.parent, exist_ok=True, parents=True)
    out_df = build_representation_vector(leiden_df,classes_df,
                                         leiden_slide_column_name=leiden_slide_column_name,
                                         leiden_leiden_column_name=leiden_leiden_column_name,
                                         classes_slide_column_name=classes_slide_column_name,
                                         classes_sample_column_name=classes_sample_column_name,
                                         classes_combined_column_name=classes_combined_column_name,
                                         classes_label_column_name=classes_label_column_name,
                                         )

    out_df.to_csv(save_to, index=False)


def filter_empty_tiles(leiden_df: pd.DataFrame,
                       classes_df: pd.DataFrame,
                       leiden_slide_column_name: str = 'combined_slides',
                       classes_combined_column_name: str = 'combined_name',):
    non_empty_df = leiden_df[leiden_slide_column_name].unique()
    return classes_df[classes_df[classes_combined_column_name].isin(non_empty_df)]


if __name__ == '__main__':
    build_csv_from_raw(leiden_csv='/Users/muang/PycharmProjects/tumour_deep_learning/data/EXT_cohort/EXT_he_combined_leiden_2p0__fold1.csv',
                       classes_csv='/Users/muang/PycharmProjects/tumour_deep_learning/data/EXT_cohort/classes.csv',
                       save_to='../datasets/external_pancancer/ext.csv')
