import torch.nn as nn


class PanCancerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=input_size,
                                              out_features=hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=output_size),
                                    nn.Softmax(dim=1))

    def forward(self, x):
        return self.layers(x)


