import functools

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from pathlib import Path

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data_manipulation.lung_caption_dataset import LungCaptionDataset
from data_manipulation.lung_caption_vocab import Vocabulary
from utils.plotting import plot_loss
from utils.training import rnn_train_model
from models.interface.LungRNN import LungRNN
from models.LSTM import LSTM
from utils.text_model_evaluate import bleu, rouge, DEFAULT_ROUGE_NAME, DEFAULT_BLEU_NAME
from utils.datadump import save_to_json
from typing import Literal

DEFAULT_DATA_FILE_PATH = Path(__file__).parent / "datasets" / "lung_text" / "TCGA_Lung_consensus.csv"
DEFAULT_VOCAB_FILE_PATH = Path(__file__).parent / "datasets" / "lung_text" / "vocab.json"
DEFAULT_SAVED_MODEL_PARENT = Path(__file__).parent / "saved_models"
DEFAULT_SAVED_MODEL_DIR_NAME = "LSTM"
DEFAULT_SAVED_MODEL_NAME = "LSTM_checkpoints"
DEFAULT_SAVED_VOCAB_NAME = "vocabulary.json"
DEFAULT_PLOT_NAME = "loss_curve"
DEFAULT_SAVED_METRIC_NAME = "metrics.json"
DEFAULT_CONFUSION_MATRIX_NAME = "confusion_matrix"

# MODEL SETTINGS
DEFAULT_INPUT_SIZE = 128    # size of feature vector
DEFAULT_EMBED_SIZE = 128    # size of embedding vector
DEFAULT_HIDDEN_SIZE = 128   # size of LSTM hidden state
DEFAULT_LSTM_STACK = 1      # stack size of LSTM

# TRAINING SETTINGS
DEFAULT_BATCH_SIZE = 32     # batch size for dataloader
DEFAULT_EPOCH = 100        # number of epochs
DEFAULT_LEARNING_RATE = 1e-5


def train_rnn(model: LungRNN,
              batch_size: int = DEFAULT_BATCH_SIZE,
              epoch: int = DEFAULT_EPOCH,
              learning_rate: float = DEFAULT_LEARNING_RATE,
              datadir: str|Path = DEFAULT_DATA_FILE_PATH,
              vocabdir: str|Path = DEFAULT_VOCAB_FILE_PATH,
              save_parent: str|Path = DEFAULT_SAVED_MODEL_PARENT,
              save_dir: str = DEFAULT_SAVED_MODEL_DIR_NAME,
              save_name: str = DEFAULT_SAVED_MODEL_NAME,
              vocab_name: str = DEFAULT_SAVED_VOCAB_NAME,
              loss_plot_name: str = DEFAULT_PLOT_NAME,
              bleu_name: str = DEFAULT_BLEU_NAME,
              rouge_name: str = DEFAULT_ROUGE_NAME,
              device:Literal["cuda", "cpu"] | None = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # load dataset
    lung_caption_dataset = LungCaptionDataset.new_from_csv(csv_path = datadir,
                                                           vocab_path = vocabdir)

    # subsampling
    # comment this out later to use the entire dataset
    # sub_lung_caption_dataset, _ = torch.utils.data.random_split(lung_caption_dataset, [0.2, 0.8])

    # train-test split
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(lung_caption_dataset, [0.7, 0.1, 0.2])

    # send to loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=lung_caption_dataset.collate_fn)

    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=lung_caption_dataset.collate_fn)

    # loss function
    # using cross entropy
    loss = nn.CrossEntropyLoss(ignore_index = lung_caption_dataset.vocab.stoi['<PAD>'])

    # optimiser
    # using adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    training_losses, validating_losses = rnn_train_model(model=model,
                                                      optimizer=optimizer,
                                                      loss_fn=loss,
                                                      train_dataloader=train_loader,
                                                      test_dataloader=validate_loader,
                                                      epochs=epoch,
                                                      device=device)


    # save model
    if not isinstance(save_parent, Path):
        save_parent = Path(save_parent)
    target_dir = save_parent / save_dir
    model.save_model_and_vocab(model_dir=target_dir/save_name,
                               vocab_dir=target_dir/vocab_name)

    # loss
    plot_loss(training_losses, validating_losses,
              save_to=target_dir/loss_plot_name,
              show_plot=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=lung_caption_dataset.collate_fn)

    # bleu and rouge
    bleu_dict = bleu(model, test_loader)
    save_to_json(bleu_dict, target_dir / bleu_name)
    rouge_dict = rouge(model, test_loader)
    save_to_json(rouge_dict, target_dir/rouge_name)



if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.load(DEFAULT_VOCAB_FILE_PATH)
    lstm_model = LSTM(input_size = DEFAULT_INPUT_SIZE,
                      embed_size = DEFAULT_EMBED_SIZE,
                      hidden_size = DEFAULT_HIDDEN_SIZE,
                      vocab = vocab,
                      num_layers = DEFAULT_LSTM_STACK)
    # lstm_model.caption(torch.Tensor([0.007932778,-0.6535041,-0.32592073,0.5959711,0.6561984,0.72325367,0.7691342,-1.0305343,-0.2527991,-0.71954805,0.18113293,-0.52599835,-0.20067441,-0.29770222,0.52636486,-0.25883275,0.3601961,-0.20289223,-0.18432501,0.023142371,-0.50373554,0.12701714,-0.36622202,-0.45415965,0.31850484,0.3967717,0.7098042,0.054774776,0.018818032,0.17783704,-0.16318966,-0.29783416,0.23743758,-0.0657612,-0.30754843,0.07342377,0.1687861,0.34562567,0.4063647,-0.3809925,0.28917754,-0.593021,-0.11398333,0.055060394,-0.71482295,0.091886915,0.35405225,-0.6085819,0.30212772,-0.6940665,0.20554326,0.24928072,-0.24627034,0.94119084,0.052024104,-0.93536055,0.6756928,-0.22274317,-0.44869697,0.51905966,-1.083598,0.70078456,-0.17058548,-0.025308877,0.19538344,-0.18474415,0.13062754,-0.21208975,0.2038778,0.58381724,-0.1663603,-0.543006,0.17167088,0.103359155,-0.8651728,0.10353868,0.6448971,0.8383773,0.92803353,0.8083539,-0.517328,0.59993535,-0.95491546,-0.23504966,-0.33204913,-0.3178196,1.2580882,-0.29586136,-0.22297817,-1.1992894,-0.61835814,-0.88648033,0.014756579,0.38000324,-0.11679,0.17192963,-0.05392863,-0.06078184,-0.3358556,0.42742553,-0.5645291,1.06095,-0.52929205,-0.5769915,-0.6159594,-0.8970743,0.8195447,-0.45337948,0.94296175,0.57874805,-0.01897861,-0.23590383,0.21447317,0.04327359,-0.18373266,0.13108721,0.7222678,-0.26989046,0.35972458,0.88928616,-0.30475873,0.3432734,0.5719182,0.257411,-0.6369363,-0.78694564,-0.10600654,-0.075793676]))
    train_rnn(model=lstm_model)