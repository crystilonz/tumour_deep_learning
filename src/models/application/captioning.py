from models.interface.LungRNN import LungRNN
from pathlib import Path
from data_manipulation.lung_caption_vocab import Vocabulary
from models.LSTM import LSTM
import pandas as pd
import torch

SCREEN_SIZE = 40  # for lines in string


class CaptionMachine:
    def __init__(self,
                 model: LungRNN,
                 dataset_path: str | Path,
                 dataset_full_name_column_name: str = 'full_names',
                 dataset_consensus_column_name: str = 'consensus',
                 dataset_start_of_vector_index: int = 14,
                 dataset_vector_length: int = 128,
                 image_hdf5: str | Path | None = None):

        self.inference_model: LungRNN = model
        self.dataset: pd.DataFrame = pd.read_csv(dataset_path)

        # accessors
        self.col_fullnames = dataset_full_name_column_name
        self.col_consensus = dataset_consensus_column_name
        self.idx_start_vector = dataset_start_of_vector_index
        self.len_vector = dataset_vector_length

        # TODO add HDF5 handling for image showing here!

    def get_by_name(self,
                    full_name: str):

        row = self.dataset.loc[self.dataset[self.col_fullnames] == full_name]
        name = row[self.col_fullnames]
        target_caption = row[self.col_consensus]
        feature = row.iloc[:, self.idx_start_vector:(self.idx_start_vector + self.len_vector)]
        return str(name.item()), str(target_caption.item()), torch.FloatTensor(feature.values)

    def caption_by_name(self,
                        full_name: str,
                        max_length = 50):

        name, target, feature_tensor = self.get_by_name(full_name)
        feature_tensor = feature_tensor.squeeze()
        caption = self.inference_model.caption(feature_tensor,
                                               max_length=max_length)
        print('Sample:', name)
        print('-' * SCREEN_SIZE)
        print('>>>> Image Captioning <<<<')
        print('Generated Caption:', self.inference_model.vocab.trim_sos_eos(caption).capitalize())
        print('Actual Caption:', target)


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
                              image_hdf5=image_hdf5)