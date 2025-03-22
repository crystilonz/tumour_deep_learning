import pickle

import numpy as np
import torch
import torchmetrics
from models.interface.LungRNN import LungRNN
from models.LSTM import LSTM
from tqdm.auto import tqdm
from pathlib import Path
from data_manipulation.lung_caption_vocab import Vocabulary
from utils.datadump import save_to_json

DEFAULT_PICKLE_PATH = Path(__file__).parent.parent / 'datasets/lung_text/z_vec.pkl'
DEFAULT_VOCAB_PATH = Path(__file__).parent.parent / 'datasets/lung_text/vocab.json'
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / 'dist_models/LSTM_z_vec/model.pt'
DEFAULT_BLEU_NAME = "bleu.json"
DEFAULT_ROUGE_NAME = "rouge.json"

def join_token_list(tok_lst):
    # this is to make joining consistent across this library
    return ' '.join(tok_lst)


def bleu(model: LungRNN,
         dataloader: torch.utils.data.DataLoader,
         max_length: int = 50) -> dict:
    bleu1 = torchmetrics.text.bleu.BLEUScore(n_gram=1)
    bleu2 = torchmetrics.text.bleu.BLEUScore(n_gram=2,
                                             weights=[0.0, 1.0])
    bleu3 = torchmetrics.text.bleu.BLEUScore(n_gram=3,
                                             weights=[0.0, 0.0, 1.0])
    bleu4 = torchmetrics.text.bleu.BLEUScore(n_gram=4,
                                             weights=[0.0, 0.0, 0.0, 1.0])
    bleu_avg = torchmetrics.text.bleu.BLEUScore(n_gram=4,
                                                weights=[0.25, 0.25, 0.25, 0.25])

    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    bleu_avg_scores = []

    print('Running BLEU metrics\n')
    model.eval()
    with torch.no_grad():
        for feature_vector, target_caption in tqdm(dataloader):
            number_of_features = feature_vector.shape[0]
            for idx in range(number_of_features):
                feature_z = feature_vector[idx, :]
                index_list = model.caption_raw_tokens(feature_z, max_length)
                pred_tok_list = model.vocab.itos_list(index_list, show_pad=False)
                pred_sentence = join_token_list(pred_tok_list)

                target_i = target_caption[idx, :]
                target_tok_list = model.vocab.itos_list(target_i, show_pad=False)
                target_sentence = join_token_list(target_tok_list)

                bleu_1_score = bleu1([pred_sentence], [[target_sentence]]).item()
                bleu_2_score = bleu2([pred_sentence], [[target_sentence]]).item()
                bleu_3_score = bleu3([pred_sentence], [[target_sentence]]).item()
                bleu_4_score = bleu4([pred_sentence], [[target_sentence]]).item()
                bleu_avg_score = bleu_avg([pred_sentence], [[target_sentence]]).item()

                bleu_1_scores.append(bleu_1_score)
                bleu_2_scores.append(bleu_2_score)
                bleu_3_scores.append(bleu_3_score)
                bleu_4_scores.append(bleu_4_score)
                bleu_avg_scores.append(bleu_avg_score)

    avg_bleu_1 = np.mean(bleu_1_scores).item()
    avg_bleu_2 = np.mean(bleu_2_scores).item()
    avg_bleu_3 = np.mean(bleu_3_scores).item()
    avg_bleu_4 = np.mean(bleu_4_scores).item()
    avg_bleu_avg = np.mean(bleu_avg_scores).item()

    sd_bleu_1 = np.std(bleu_1_scores).item()
    sd_bleu_2 = np.std(bleu_2_scores).item()
    sd_bleu_3 = np.std(bleu_3_scores).item()
    sd_bleu_4 = np.std(bleu_4_scores).item()
    sd_bleu_avg = np.std(bleu_avg_scores).item()

    results_dict = {
        'bleu-1': avg_bleu_1,
        'bleu-2': avg_bleu_2,
        'bleu-3': avg_bleu_3,
        'bleu-4': avg_bleu_4,
        'bleu-avg': avg_bleu_avg,

        'bleu-1_sd': sd_bleu_1,
        'bleu-2_sd': sd_bleu_2,
        'bleu-3_sd': sd_bleu_3,
        'bleu-4_sd': sd_bleu_4,
        'bleu-avg_sd': sd_bleu_avg,
    }

    return results_dict


def rouge(model: LungRNN,
          dataloader: torch.utils.data.DataLoader,
          max_length: int = 50) -> dict:

    # rouge_metric = torchmetrics.text.rouge.ROUGEScore(rouge_keys=('rouge1', 'rouge2', 'rougeL'))
    rouge_metric = lambda pred, target: torchmetrics.functional.text.rouge.rouge_score(pred, target, rouge_keys=('rouge1', 'rouge2', 'rougeL'), normalizer=lambda x: x)

    rouge_dict = {
        'rouge1_fscore_list': [],
        'rouge1_precision_list': [],
        'rouge1_recall_list': [],
        'rouge2_fscore_list': [],
        'rouge2_precision_list': [],
        'rouge2_recall_list': [],
        'rougeL_fscore_list': [],
        'rougeL_precision_list': [],
        'rougeL_recall_list': []
    }
    print('Running ROUGE metrics.\n')
    model.eval()
    with torch.no_grad():
        for feature_vector, target_caption in tqdm(dataloader):
            number_of_features = feature_vector.shape[0]
            for idx in range(number_of_features):
                feature_z = feature_vector[idx, :]
                index_list = model.caption_raw_tokens(feature_z, max_length)
                pred_tok_list = model.vocab.itos_list(index_list, show_pad=False)
                pred_sentence = join_token_list(pred_tok_list)

                target_i = target_caption[idx, :]
                target_tok_list = model.vocab.itos_list(target_i, show_pad=False)
                target_sentence = join_token_list(target_tok_list)

                rouge_eval = rouge_metric(pred_sentence, target_sentence)

                rouge_dict['rouge1_fscore_list'].append(rouge_eval['rouge1_fmeasure'].item())
                rouge_dict['rouge1_precision_list'].append(rouge_eval['rouge1_precision'].item())
                rouge_dict['rouge1_recall_list'].append(rouge_eval['rouge1_recall'].item())
                rouge_dict['rouge2_fscore_list'].append(rouge_eval['rouge2_fmeasure'].item())
                rouge_dict['rouge2_precision_list'].append(rouge_eval['rouge2_precision'].item())
                rouge_dict['rouge2_recall_list'].append(rouge_eval['rouge2_recall'].item())
                rouge_dict['rougeL_fscore_list'].append(rouge_eval['rougeL_fmeasure'].item())
                rouge_dict['rougeL_precision_list'].append(rouge_eval['rougeL_precision'].item())
                rouge_dict['rougeL_recall_list'].append(rouge_eval['rougeL_recall'].item())

    rouge_results = dict()
    rouge_results['rouge1_fscore_mean'] = np.mean(rouge_dict['rouge1_fscore_list']).item()
    rouge_results['rouge1_precision_mean'] = np.mean(rouge_dict['rouge1_precision_list']).item()
    rouge_results['rouge1_recall_mean'] = np.mean(rouge_dict['rouge1_recall_list']).item()
    rouge_results['rouge2_fscore_mean'] = np.mean(rouge_dict['rouge2_fscore_list']).item()
    rouge_results['rouge2_precision_mean'] = np.mean(rouge_dict['rouge2_precision_list']).item()
    rouge_results['rouge2_recall_mean'] = np.mean(rouge_dict['rouge2_recall_list']).item()
    rouge_results['rougeL_fscore_mean'] = np.mean(rouge_dict['rougeL_fscore_list']).item()
    rouge_results['rougeL_precision_mean'] = np.mean(rouge_dict['rougeL_precision_list']).item()
    rouge_results['rougeL_recall_mean'] = np.mean(rouge_dict['rougeL_recall_list']).item()

    rouge_results['rouge1_fscore_sd'] = np.std(rouge_dict['rouge1_fscore_list']).item()
    rouge_results['rouge1_precision_sd'] = np.std(rouge_dict['rouge1_precision_list']).item()
    rouge_results['rouge1_recall_sd'] = np.std(rouge_dict['rouge1_recall_list']).item()
    rouge_results['rouge2_fscore_sd'] = np.std(rouge_dict['rouge2_fscore_list']).item()
    rouge_results['rouge2_precision_sd'] = np.std(rouge_dict['rouge2_precision_list']).item()
    rouge_results['rouge2_recall_sd'] = np.std(rouge_dict['rouge2_recall_list']).item()
    rouge_results['rougeL_fscore_sd'] = np.std(rouge_dict['rougeL_fscore_list']).item()
    rouge_results['rougeL_precision_sd'] = np.std(rouge_dict['rougeL_precision_list']).item()
    rouge_results['rougeL_recall_sd'] = np.std(rouge_dict['rougeL_recall_list']).item()

    return rouge_results

def pickle_test_lstm(input_size: int = 128,
                     embed_size: int = 128,
                     hidden_size: int = 128,
                     num_layers: int = 1,
                     vocab_path: str|Path = DEFAULT_VOCAB_PATH,
                     pickle_path: str|Path = DEFAULT_PICKLE_PATH,
                     model_path: str|Path = DEFAULT_MODEL_PATH,
                     results_dir: str|Path = None,
                     bleu_name: str = DEFAULT_BLEU_NAME,
                     rouge_name: str = DEFAULT_ROUGE_NAME
                     ):
    with open(pickle_path, 'rb') as f:
        split_dict = pickle.load(f)

    if results_dir is None:
        results_dir = Path(model_path).parent

    test_dataset = split_dict['testing_dataset']
    collate_fn = split_dict['collate_fn']

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=collate_fn)

    # build model
    vocab = Vocabulary()
    vocab.load(vocab_path)
    model = LSTM(input_size=input_size,
                 embed_size=embed_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 vocab=vocab)

    print("Loading model...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Model loaded.")

    print("Calculating Bleu...")
    bleu_dict = bleu(model, test_loader)
    print("Bleu calculated.")

    print("Calculating ROUGE...")
    rouge_dict = rouge(model, test_loader)
    print("ROUGE calculated.")

    print(f"BLEU: {bleu_dict['bleu-avg']}")
    print(f"ROUGE-1: {rouge_dict['rouge1_recall_mean']}")
    print(f"ROUGE-2: {rouge_dict['rouge2_recall_mean']}")
    print(f"ROUGE-L: {rouge_dict['rougeL_recall_mean']}")

    print(f"Saving to {results_dir}.")
    save_to_json(bleu_dict, results_dir / bleu_name)
    save_to_json(rouge_dict, results_dir / rouge_name)







