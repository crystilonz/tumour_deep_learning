import spacy
import pandas as pd
import torch
import copy
from typing import Literal
from pathlib import Path
import json
from utils.files_handling import dict_json_pretty

# special vocab list
START_TOK = '<SOS>'
END_TOK = '<EOS>'
_special_itos = {0: '<PAD>', 1: START_TOK, 2: END_TOK, 3: '<UNK>'}  # only change this one
_special_stoi = {s: i for i, s in _special_itos.items()}
_start_index = len(_special_itos)

class Vocabulary:
    def __init__(self, min_freq=1, max_size=None, tokenizer='spacy'):
        # number to string
        self.itos = copy.deepcopy(_special_itos)    # special vocabs

        # string to number
        self.stoi = copy.deepcopy(_special_stoi)

        # length of vocab
        self.length = _start_index

        # min frequency
        if min_freq < 1:
            min_freq = 1
        self.min_freq = min_freq

        self.max_size = max_size
        self.tokenizer_text = tokenizer
        self.tokenizer = Vocabulary.resolve_tokenizer(tokenizer)

    @staticmethod
    def resolve_tokenizer(tok):
        if tok == 'spacy':
            return spacy.load('en_core_web_md')
        else:
            return spacy.blank('en')


    def build_vocab(self, texts: list[str]):
        freq = {}

        for text in texts:
            for token in self.tokenize(text):
                if token not in freq:
                    freq[token] = 1
                else:
                    freq[token] += 1

        if self.max_size is not None:
            if len(freq) > self.max_size:
                # sort
                freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        for token in freq.keys():
            if self.max_size is not None and self.length > self.max_size:
                # over capacity
                return

            if freq[token] < self.min_freq:
                # does not reach threshold -> pruned
                continue

            # add to vocab
            self.itos[self.length] = token
            self.stoi[token] = self.length
            self.length += 1


    def numericalize(self, text: str, output:Literal['list', 'torch'] = 'torch'):
        tokens = self.tokenize(text)
        idx_list = [self.stoi[t]
                    if t in self.stoi else self.stoi['<UNK>']
                    for t in tokens]
        if output == 'list':
            return idx_list
        elif output == 'torch':
            return torch.LongTensor(idx_list)
        else:
            # wrong specs
            # default to torch
            return torch.FloatTensor(idx_list)

    def tokenize(self, text):
        return [token.text.lower() for token in self.tokenizer(text)]

    def save(self, path:str|Path):
        save_json = {
            'stoi': self.stoi,
            'min_freq': self.min_freq,
            'max_size': self.max_size,
            'tokenizer': self.tokenizer_text,
        }

        if not isinstance(path, Path):
            path = Path(path)

        dict_json_pretty(save_json, path)


    def load(self, path:str|Path):
        with open(path, 'r') as f:
            json_dict = json.load(f, parse_int=int)

        self.stoi = json_dict['stoi']
        self.itos = {i: s for s, i in self.stoi.items()}
        self.length = len(self.stoi)
        self.min_freq = json_dict['min_freq']
        self.max_size = json_dict['max_size']
        self.tokenizer_text = json_dict['tokenizer']
        self.tokenizer = Vocabulary.resolve_tokenizer(json_dict['tokenizer'])

    def translate_from_index_list(self, lst: torch.Tensor|list, show_pad=False):
        lst_tokens = self.itos_list(lst, show_pad)
        return ' '.join(lst_tokens).replace(' ,', ',').replace(' .', '.')


    def itos_list(self, lst: torch.Tensor|list, show_pad=False):
        if isinstance(lst, torch.Tensor):
            lst = lst.tolist()
        if show_pad:
            lst_tokens = [self.itos[tok] for tok in lst]
        else:
            lst_tokens = [self.itos[tok] for tok in lst if tok != self.stoi['<PAD>']]
        return lst_tokens

    def trim_sos_eos(self, s:str):
        # left
        while True:
            old_string = s
            s = s.lstrip(START_TOK).lstrip(' ')
            if old_string == s:
                left_trimmed_s = s
                break

        # right
        while True:
            old_string = left_trimmed_s
            left_trimmed_s = left_trimmed_s.rstrip(END_TOK).lstrip(' ')
            if old_string == left_trimmed_s:
                trimmed_s = left_trimmed_s
                break

        return trimmed_s




def extract_text_list_from_consensus(csv_file: Path|str, col_name:str='consensus'):
    if isinstance(csv_file, str):
        csv_file = Path(csv_file)

    with open(csv_file, 'r') as f:
        cs_df = pd.read_csv(f)

    return cs_df[col_name].tolist()


if __name__ == '__main__':
    consensus_list = extract_text_list_from_consensus(Path(__file__).parent.parent / 'datasets/lung_text/consensus.csv')
    vocab = Vocabulary()
    vocab.build_vocab(consensus_list)
    print(vocab.itos)