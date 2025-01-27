import pandas as pd
import torch
from data_manipulation.lung_caption_vocab import Vocabulary, extract_text_list_from_consensus
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from overrides import override
from pathlib import Path

class LungCaptionDataset(Dataset):
    def __init__(self,dataframe:pd.DataFrame,
                 vocab:Vocabulary,
                 caption_col_name:str,
                 vector_idx:int,
                 vector_len:int|None = None):

        if vector_len is None:
            vector_df = dataframe.iloc[:, vector_idx:]
        else:
            vector_df = dataframe.iloc[:, vector_idx:(vector_idx + vector_len - 1)]

        self.caption_series:pd.Series = dataframe[caption_col_name]
        self.vector_tensor:torch.Tensor = torch.FloatTensor(vector_df.values())
        self.vocab:Vocabulary = vocab

    @override
    def __len__(self):
        return len(self.caption_series)

    @override
    def __getitem__(self, idx):
        feature_vector = self.vector_tensor[idx]
        caption = self.caption_series[idx]

        caption_vector = [self.vocab.stoi('<SOS>')] + [self.vocab.numericalize(caption, 'list')] + [self.vocab.stoi('<EOS>')]
        caption_tensor = torch.LongTensor(caption_vector)
        return feature_vector, caption_tensor

    def collate_fn(self, batch):
        features_list = [item[0] for item in batch]
        features_batch = torch.stack(features_list)

        captions_list = [item[1] for item in batch]
        captions_batch = pad_sequence(captions_list,
                                      batch_first=True,
                                      padding_value=self.vocab.stoi('<PAD>'),
                                      padding_side='right')

        return features_batch, captions_batch

    @staticmethod
    def new_from_csv(csv_path:Path|str,
                       vocab_path:Path|str,
                       caption_col_name:str = 'consensus',
                       vector_idx:int = 14,
                       vector_len:int|None = None,
                       ):

        lung_df = pd.read_csv(csv_path)
        vocab = Vocabulary()
        vocab.load(vocab_path)

        return LungCaptionDataset(lung_df,
                                  vocab=vocab,
                                  caption_col_name=caption_col_name,
                                  vector_idx=vector_idx,
                                  vector_len=vector_len)


