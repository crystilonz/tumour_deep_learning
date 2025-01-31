import h5py

from models.interface.LungRNN import LungRNN
from pathlib import Path
from data_manipulation.lung_caption_vocab import Vocabulary
from models.LSTM import LSTM
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Literal

SCREEN_SIZE = 40  # for lines in string


class CaptionMachine:
    def __init__(self,
                 model: LungRNN,
                 dataset_path: str | Path,
                 dataset_full_name_column_name: str = 'full_names',
                 dataset_consensus_column_name: str = 'consensus',
                 dataset_start_of_vector_index: int = 14,
                 dataset_vector_length: int = 128,
                 image_hdf5: str | Path | None = None,
                 hdf5_image_field: str = 'img',
                 hdf5_name_field: list[str] | tuple[str] | str = 'samples',
                 hdf5_name_concat: str = '_',
                 hdf5_filter_field: str | None = 'labels',
                 hdf5_filter_value: list | tuple = (0.0, 1.0, 2.0, 3.0, 4.0)):

        self.inference_model: LungRNN = model
        self.dataset: pd.DataFrame = pd.read_csv(dataset_path)

        # accessors
        self.col_fullnames = dataset_full_name_column_name
        self.col_consensus = dataset_consensus_column_name
        self.idx_start_vector = dataset_start_of_vector_index
        self.len_vector = dataset_vector_length

        if image_hdf5 is not None:
            self.show_image = True
            self.hdf5_file_path = image_hdf5
            with h5py.File(image_hdf5, 'r') as f:
                # resolve full name
                if isinstance(hdf5_name_field, list) or isinstance(hdf5_name_field, tuple):
                    columns = [pd.DataFrame(np.array(f[col_name]), dtype='string') for col_name in hdf5_name_field]
                    concatenated = pd.concat(columns, axis=1)
                    self.hdf5_complete_name = concatenated.apply(lambda x: hdf5_name_concat.join(x), axis=1)
                else:
                    self.hdf5_complete_name = pd.Series(np.array(f[hdf5_name_field]), dtype='string')

                # extract filter
                if hdf5_filter_field is not None:
                    self.filtering = True
                    self.filter_series = pd.Series(np.array(f[hdf5_filter_field]))
                else:
                    self.filtering = False

            hdf5_length = len(self.hdf5_complete_name)
            original_indices = pd.DataFrame(np.arange(hdf5_length), columns=['original_indices'])
            renamed_complete_name_pd = pd.DataFrame(self.hdf5_complete_name, columns=['full_names'])
            self.hdf5_complete_name = pd.concat([original_indices, renamed_complete_name_pd], axis=1)

            # filter
            if self.filtering:
                self.hdf5_complete_name = self.hdf5_complete_name[self.filter_series.isin(hdf5_filter_value)]

            self.hdf5_image_field = hdf5_image_field
        else:
            self.show_image = False

    def get_hdf5_index_by_name(self,
                               full_name):
        if not self.show_image:
            print("No HDF5 file.")
            return None

        indices = self.hdf5_complete_name[self.hdf5_complete_name['full_names'] == full_name]['original_indices']

        if len(indices) == 0:
            print("No match in HDF5 file.")
            return None
        elif len(indices) == 1:
            return int(indices.item())
        else:
            # this should never happen unless file is corrupted or process pipe is wrong somehow
            print("Multiple matches in HDF5 file.")
            return indices

    def get_image_by_index(self,
                           index):
        with h5py.File(self.hdf5_file_path, 'r') as f:
            img = np.array(f[self.hdf5_image_field][index])
        return img

    def get_image_by_name(self,
                          full_name: str):
        index = self.get_hdf5_index_by_name(full_name)
        if index is None:
            return None
        img = self.get_image_by_index(index)
        return img

    def get_random_hdf5(self):
        if self.show_image:
            random_row = self.hdf5_complete_name.sample()
            return random_row['full_names'].item(), int(random_row['original_indices'].item())
        else:
            return None

    def get_random_dataset(self):
        random_dataset_row = self.dataset.sample()
        return random_dataset_row[self.col_fullnames].item()

    def get_by_name(self,
                    full_name: str):

        row = self.dataset.loc[self.dataset[self.col_fullnames] == full_name]
        name = row[self.col_fullnames]
        target_caption = row[self.col_consensus]
        feature = row.iloc[:, self.idx_start_vector:(self.idx_start_vector + self.len_vector)]
        return str(name.item()), str(target_caption.item()), torch.FloatTensor(feature.values)

    def name_is_in_hdf5(self, full_name):
        return (self.hdf5_complete_name['full_names'] == full_name).any()

    def name_is_in_dataset(self, full_name):
        return (self.dataset[self.col_fullnames] == full_name).any()

    def caption_by_name(self,
                        full_name: str,
                        max_length=50,
                        show_img=True,
                        swap_rgb=False):

        name, target, feature_tensor = self.get_by_name(full_name)
        feature_tensor = feature_tensor.squeeze()
        caption = self.inference_model.caption(feature_tensor,
                                               max_length=max_length)
        print('Sample:', name)

        # show image
        if self.show_image == True and show_img == True:
            img = self.get_image_by_name(name)
            if img is None:
                print(f'Cannot find image for sample {name}.')
            else:
                if swap_rgb:
                    img = img[:, :, [2, 1, 0]]
                plt.imshow(img)
                plt.axis('off')
                plt.show()

        print('-' * SCREEN_SIZE)
        print('>>>> Image Captioning <<<<')
        print('Generated Caption:', self.inference_model.vocab.trim_sos_eos(caption).capitalize())
        print('Actual Caption:', target)

    def caption_random_image(self,
                             max_length=50,
                             show_img=True,
                             source: Literal['hdf5', 'dataset'] = 'dataset',
                             swap_rgb=False):
        if source == 'hdf5':
            if not self.show_image:
                print("No HDF5 file.")
                return
            while True:
                random_name_in_hdf5, _ = self.get_random_hdf5()
                if self.name_is_in_dataset(random_name_in_hdf5):
                    break
                print(f'Sample {random_name_in_hdf5} not found in dataset. Randomising a new one...')
            self.caption_by_name(random_name_in_hdf5, max_length, show_img, swap_rgb)
        elif source == 'dataset':
            while True:
                random_name_in_dataset = self.get_random_dataset()
                if show_img:
                    # check hdf
                    if not self.show_image:
                        print("No HDF5 file, no image to be shown.")
                        show_img = False
                        break
                    else:
                        # check existence
                        if self.name_is_in_hdf5(random_name_in_dataset):
                            break
                        print(f'Sample {random_name_in_dataset} not found in hdf5. Randomising a new one...')
            self.caption_by_name(random_name_in_dataset, max_length, show_img, swap_rgb)
        else:
            raise ValueError('Source must be either hdf5 or dataset.')

    @staticmethod
    def create_LSTM(input_size,
                    embed_size,
                    hidden_size,
                    vocab_path,
                    checkpoints_path,
                    dataset_path: str | Path,
                    dataset_full_name_column_name: str = 'full_names',
                    dataset_consensus_column_name: str = 'consensus',
                    dataset_start_of_vector_index: int = 14,
                    dataset_vector_length: int = 128,
                    image_hdf5: str | Path | None = None,
                    hdf5_image_field: str = 'img',
                    hdf5_name_field: list[str] | str = 'samples',
                    hdf5_name_concat: str = '_',
                    hdf5_filter_field: str | None = 'labels',
                    hdf5_filter_value: list | tuple = (0.0, 1.0, 2.0, 3.0, 4.0),
                    num_layers=1):

        model, vocab = LSTM.load_model_from_files(input_size=input_size,
                                                  embed_size=embed_size,
                                                  hidden_size=hidden_size,
                                                  vocab_dir=vocab_path,
                                                  checkpoints_dir=checkpoints_path,
                                                  num_layers=num_layers)

        return CaptionMachine(model=model,
                              dataset_path=dataset_path,
                              dataset_full_name_column_name=dataset_full_name_column_name,
                              dataset_consensus_column_name=dataset_consensus_column_name,
                              dataset_start_of_vector_index=dataset_start_of_vector_index,
                              dataset_vector_length=dataset_vector_length,
                              image_hdf5=image_hdf5,
                              hdf5_image_field=hdf5_image_field,
                              hdf5_name_field=hdf5_name_field,
                              hdf5_name_concat=hdf5_name_concat,
                              hdf5_filter_field=hdf5_filter_field,
                              hdf5_filter_value=hdf5_filter_value)
