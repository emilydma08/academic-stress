import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim=1, num_layers=1, dropout_rate=0):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
            )

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
            )

        self.output_layer = nn.Linear(
            self.hidden_dim,
            4
        )


    def forward(self, x):
      a = self.input_layer(x)
      for _ in range(self.num_layers):
        a = self.hidden_layer(a)
      a = self.output_layer(a)
      return a