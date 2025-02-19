import pickle
import torch
import numpy as np
import random
from pathlib import Path
from data_manipulation.lung_caption_vocab import Vocabulary
from data_manipulation.lung_caption_dataset import LungCaptionDataset

DEFAULT_NAME = 'splits.pkl'
SEED = 5

def split_lung_caption(csv_source: str | Path,
                       vocab_path: str | Path,
                       splits: list[float],
                       save_to: str | Path = None):
    csv_source = Path(csv_source)
    vocab_path = Path(vocab_path)
    lung_caption_dataset = LungCaptionDataset.new_from_csv(csv_path=csv_source,
                                                           vocab_path=vocab_path)
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # split
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(lung_caption_dataset,
                                                                                  splits)
    collate_fn = lung_caption_dataset.collate_fn

    # dict to save
    d = {'training_dataset': train_dataset,
         'validating_dataset': validate_dataset,
         'testing_dataset': test_dataset,
         'collate_fn': collate_fn}

    # pickle
    if save_to is None:
        # parent
        save_dir = csv_source.parent / DEFAULT_NAME
        with open(save_dir, 'wb') as f:
            pickle.dump(d, f)
    else:
        with open(save_to, 'wb') as f:
            pickle.dump(d, f)


if __name__ == '__main__':
    split_lung_caption(csv_source='/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/TCGA_Lung_consensus.csv',
                       vocab_path='/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/vocab.json',
                       splits=[0.7, 0.1, 0.2],
                       save_to='/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/z_vec.pkl')
    split_lung_caption(csv_source='/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/TCGA_Lung_consensus_h_vec.csv',
                       vocab_path='/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/vocab.json',
                       splits=[0.7, 0.1, 0.2],
                       save_to='/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/h_vec.pkl')
    
