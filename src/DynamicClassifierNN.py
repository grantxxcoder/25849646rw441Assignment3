
import torch
import torch.nn as nn
class DynamicMLP(nn.Module):
    def __init__(self, input_dim, activation=nn.ReLU):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Identity())
        self.output_layer = nn.Linear(input_dim, 2).to(self.device)
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
            new_layer = nn.Linear(self.input_dim, 2).to(self.device)
            self.hidden_layers[layer_idx] = new_layer
            self.output_layer = nn.Linear(2, self.output_layer.out_features).to(self.device)
        else:
            in_features = old_layer.in_features
            current_out = old_layer.out_features
            if current_out < 512:
                next_out = min(current_out * 2, 512)
                new_layer = nn.Linear(in_features, next_out).to(self.device)
                self.hidden_layers[layer_idx] = new_layer
            else:
                self.add_hidden_layer(1)
                self.current_layer += 1 
                return
        
        last_out = self.hidden_layers[-1].out_features if not isinstance(self.hidden_layers[-1], nn.Identity) else self.input_dim
        self.output_layer = nn.Linear(last_out, self.output_layer.out_features).to(self.device)

        print(f"Added neurons to layer {layer_idx}, new size: {self.hidden_layers[layer_idx].out_features}")
        print(f"Output layer updated to {self.output_layer.in_features} -> {self.output_layer.out_features}")
        
    def add_hidden_layer(self, n_neurons):
        if not isinstance(self.hidden_layers[-1], nn.Identity):
            prev_dim = self.hidden_layers[-1].out_features
        else:
            prev_dim = self.input_dim
        self.hidden_layers.append(nn.Linear(prev_dim, n_neurons).to(self.device))
        self.output_layer = nn.Linear(n_neurons, self.output_layer.out_features).to(self.device)

    def update_output_layer(self, new_output_dim):
        # if not isinstance(self.hidden_layers[-1], nn.Identity):
        #     prev_dim = self.hidden_layers[-1].out_features
        # else:
        #     prev_dim = self.input_dim
        self.output_layer = nn.Linear(self.output_layer.in_features, new_output_dim).to(self.device)

    
    def print_structure(self):
        print("Current MLP Structure:")
        for idx, layer in enumerate(self.hidden_layers):
            layer_type = type(layer).__name__
            if isinstance(layer, nn.Identity):
                print(f" Layer {idx}: Identity (no neurons)")
            else:
                print(f" Layer {idx}: {layer_type} with {layer.out_features} neurons")
        print(f" Output Layer: Linear with {self.output_layer.out_features} output classes")