import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation='relu'):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        if activation=='relu':
            activation_function = nn.ReLU
        else:
            activation_function = nn.Tanh
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_function())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)