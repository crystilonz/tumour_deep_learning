import torch.nn as nn


class LeakyPanCancerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=input_size,
                                              out_features=hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size,
                                              out_features=output_size)
                                    )

    def forward(self, x):
        return self.layers(x)
