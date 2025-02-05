import numpy as np
import torch
import torchmetrics
from models.interface.LungRNN import LungRNN
from tqdm.auto import tqdm


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
    rouge_metric = lambda pred, target: torchmetrics.functional.text.rouge.rouge_score(pred, target, rouge_keys=('rouge1', 'rouge2', 'rougeL'))

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