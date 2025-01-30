import torch.nn as nn
import torch
from anyio.lowlevel import checkpoint

from data_manipulation.lung_caption_vocab import Vocabulary
from utils.training import save_model
from models.interface.LungRNN import LungRNN


class LSTM(LungRNN):
    def __init__(self, input_size, embed_size, hidden_size, vocab:Vocabulary, num_layers=1):
        super(LSTM, self).__init__()
        self._vocab = vocab
        self.feature_linear = nn.Linear(input_size, embed_size)
        self.embedding = nn.Embedding(self.vocab.length, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab.length)

    @property
    def vocab(self):
        return self._vocab

    def forward(self, features, captions):
        # remove <EOS> from caption
        feature_embed = self.feature_linear(features)  # (N, input_size) --> (N, embed_size)
        caption_embed = self.embedding(captions[:, :-1])  # (N, caption_len - 1) --> (N, caption_len - 1, embed_size)

        # feature_embed: (N, embed_size)
        # feature_embed.unsqueeze(dim=1): (N, 1, embed_size)
        # stack on top of the caption embed

        embed = torch.cat((feature_embed.unsqueeze(dim=1), caption_embed), dim=1)  # (N, caption_len, hidden_size)

        # lstm_out: (caption_len, N, hidden_size)
        lstm_out, _ = self.lstm(embed)
        output = self.linear(lstm_out)
        return output  # (N, caption_len, vocab_size)

    def caption(self, features, max_length=50):
        caption = []
        lstm_state = None

        self.eval()
        with torch.no_grad():
            embed = self.feature_linear(features).unsqueeze(dim=0)  # (1, input_size)
            for _ in range(max_length):
                lstm_out, lstm_state = self.lstm(embed, lstm_state)  # lstm_out: (1, hidden_size)
                out = self.linear(lstm_out)  # out: (1, vocab_size)
                prediction = out.argmax(dim=1)
                caption.append(prediction.item())
                embed = self.embedding(prediction)

                if self.vocab.itos[prediction.item()] == '<EOS>':
                    break

        return self.vocab.translate_from_index_list(caption)

    def save_model_and_vocab(self, model_dir, vocab_dir):
        save_model(self, model_dir)
        self.vocab.save(vocab_dir)

    @staticmethod
    def load_model_from_files(input_size,
                              embed_size,
                              hidden_size,
                              vocab_dir,
                              checkpoints_dir = None,
                              num_layers=1) -> (LungRNN, Vocabulary):

        loaded_vocab = Vocabulary()
        loaded_vocab.load(vocab_dir)
        model = LSTM(input_size, embed_size, hidden_size, loaded_vocab, num_layers)
        if checkpoints_dir is not None:
            model.load_state_dict(torch.load(checkpoints_dir, weights_only=True))

        return model, loaded_vocab
