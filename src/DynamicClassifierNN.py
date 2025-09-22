import torch.nn as nn

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = activation()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

    def add_hidden_neurons(self, layer_idx, n_new_neurons):
        # Add neurons to an existing hidden layer
        old_layer = self.hidden_layers[layer_idx]
        in_features = old_layer.in_features
        out_features = old_layer.out_features + n_new_neurons
        new_layer = nn.Linear(in_features, out_features)
        self.hidden_layers[layer_idx] = new_layer

    def add_hidden_layer(self, n_neurons):
        # Add a new hidden layer at the end
        if self.hidden_layers:
            prev_dim = self.hidden_layers[-1].out_features
        else:
            prev_dim = self.input_dim
      
        self.hidden_layers.append(nn.Linear(prev_dim, n_neurons))
        self.output_layer = nn.Linear(n_neurons, self.output_layer.out_features)

    def update_output_layer(self, new_output_dim):
        # Add new output neuron for new class
        if self.hidden_layers:
            prev_dim = self.hidden_layers[-1].out_features
        else:
            prev_dim = self.input_dim
     
        self.output_layer = nn.Linear(prev_dim, new_output_dim)