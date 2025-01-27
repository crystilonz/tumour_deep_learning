import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTM, self).__init__()
        self.feature_linear = nn.Linear(input_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # remove <EOS> from caption
        feature_embed = self.feature_linear(features)                                       # (N, input_size) --> (N, embed_size)
        caption_embed = self.embedding(captions[:, :-1])                                    # (N, caption_len - 1) --> (N, caption_len - 1, embed_size)

        # feature_embed: (N, embed_size)
        # feature_embed.unsqueeze(dim=1): (N, 1, embed_size)
        # stack on top of the caption embed

        embed = torch.cat((feature_embed.unsqueeze(dim=1), caption_embed), dim=1)   # (N, caption_len, embed_size)

        # hn: (num_layers, N, hidden_size)
        hn, cn = self.lstm(embed)
        output = self.linear(hn[-1])  # take the last hidden layer in the stack only

        # (N, vocab_size)
        return output

    def caption(self, features, vocab, max_length=30):
        caption = []
        cn = None
        features.squeeze()

        self.eval()
        with torch.no_grad():
            embed = self.feature_linear(features)                                           # (input_size)
            for _ in range(max_length):
                hn, cn = self.lstm(embed, cn)                                               # hn: (num_layers, hidden_size)
                out = self.linear(hn[-1])                                                   # hn[-1]: (hidden_size), squeeze just in case
                prediction = out.argmax()
                caption.append(prediction.item())
                embed = self.embedding(prediction)

                if vocab.itos[prediction.item()] == '<EOS>':
                    break

        token_list = [vocab.itox[i] for i in caption]
        return ' '.join(token_list).replace(' ,', ',').replace(' .', '.')