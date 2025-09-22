import torch.nn as nn

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Identity())
        self.output_layer = nn.Linear(input_dim, 2)
        self.activation = activation()
        self.current_layer = 0 

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x)) if not isinstance(layer, nn.Identity) else layer(x)
        return self.output_layer(x)

    def add_hidden_neurons(self):
        # Add neurons to the current hidden layer in powers of 2
        layer_idx = self.current_layer
        old_layer = self.hidden_layers[layer_idx]
        if isinstance(old_layer, nn.Identity):
            new_layer = nn.Linear(self.input_dim, 1)
            self.hidden_layers[layer_idx] = new_layer
            self.output_layer = nn.Linear(1, self.output_layer.out_features)
        else:
            in_features = old_layer.in_features
            current_out = old_layer.out_features
            if current_out < 512:
                next_out = min(current_out * 2, 512)
                new_layer = nn.Linear(in_features, next_out)
                self.hidden_layers[layer_idx] = new_layer
                if layer_idx == len(self.hidden_layers) - 1:
                    self.output_layer = nn.Linear(next_out, self.output_layer.out_features)
            else:
                self.add_hidden_layer(1)
                self.current_layer += 1 

    def add_hidden_layer(self, n_neurons):
        if not isinstance(self.hidden_layers[-1], nn.Identity):
            prev_dim = self.hidden_layers[-1].out_features
        else:
            prev_dim = self.input_dim
        self.hidden_layers.append(nn.Linear(prev_dim, n_neurons))
        self.output_layer = nn.Linear(n_neurons, self.output_layer.out_features)

    def update_output_layer(self, new_output_dim):
        if not isinstance(self.hidden_layers[-1], nn.Identity):
            prev_dim = self.hidden_layers[-1].out_features
        else:
            prev_dim = self.input_dim
        self.output_layer = nn.Linear(prev_dim, new_output_dim)