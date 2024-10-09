from torch import nn

class PanCancerLogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(self.linear(x))

