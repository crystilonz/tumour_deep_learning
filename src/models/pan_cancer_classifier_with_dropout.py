import torch.nn as nn


class PanCancerClassifierWithDropout(nn.Module):
    """Same as PanCancerClassifier but introduced dropouts (0.2) in between layers"""
    def __init__(self, input_size, hidden_size, output_size, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=input_size,
                                              out_features=hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=output_size),
                                    nn.Softmax(dim=1))

    def forward(self, x):
        return self.layers(x)


