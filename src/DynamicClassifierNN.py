
from doctest import debug
import torch
import torch.nn as nn
import numpy as np
import random
class DynamicMLP(nn.Module):
    def __init__(self, input_dim, max_layers=3, activation=nn.ReLU, seed=42, debug=False):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed) 
        torch.manual_seed(seed)  
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed) 
            torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Identity())
        self.output_layer = nn.Linear(input_dim, 2).to(self.device)
        self.activation = activation()
        self.current_layer = 0 
        self.max_layers = max_layers
        self.debug = debug

    def _expand_output_layer(self, new_in_features=None, new_out_features=None):
        """Expand the output layer to new input/out dims while preserving any overlapping weights.
        - new_in_features: int or None (if None, keep current in_features)
        - new_out_features: int or None (if None, keep current out_features)
        Copies the overlapping block of weights and biases from the old output layer into the new one.
        """
        old = self.output_layer
        in_f = new_in_features if new_in_features is not None else old.in_features
        out_f = new_out_features if new_out_features is not None else old.out_features
        new_layer = nn.Linear(in_f, out_f).to(self.device)
        # Copy overlapping weights/biases
        with torch.no_grad():
            min_out = min(old.out_features, out_f)
            min_in = min(old.in_features, in_f)
            # zero new weights then copy the overlapping block
            new_layer.weight.zero_()
            new_layer.bias.zero_()
            new_layer.weight[:min_out, :min_in] = old.weight[:min_out, :min_in]
            new_layer.bias[:min_out] = old.bias[:min_out]
        self.output_layer = new_layer

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
            # expand output layer input dim to match new last_out while preserving existing weights
            self._expand_output_layer(new_in_features=2)
        else:
            in_features = old_layer.in_features
            current_out = old_layer.out_features
            if current_out < 512:
                next_out = min(current_out * 2, 512)
                new_layer = nn.Linear(in_features, next_out).to(self.device)
                self.hidden_layers[layer_idx] = new_layer
            else:
                new_layer_added = self.add_hidden_layer(1)
                if not new_layer_added:
                    if self.debug:
                        print("Cannot add more layers or neurons.")
                    return False
                self.current_layer += 1 
                return True

        # update output layer to accept the new last hidden layer size (preserve weights)
        last_out = self.hidden_layers[-1].out_features if not isinstance(self.hidden_layers[-1], nn.Identity) else self.input_dim
        self._expand_output_layer(new_in_features=last_out)
        if self.debug:
            print(f"Added neurons to layer {layer_idx}, new size: {self.hidden_layers[layer_idx].out_features}")
            print(f"Output layer updated to {self.output_layer.in_features} -> {self.output_layer.out_features}")
        return False
    
    def add_hidden_layer(self, n_neurons):
        # have a maximum amount of layers
        if len(self.hidden_layers) >= self.max_layers:
            if self.debug:
                print("Maximum number of hidden layers reached.")
            return False
        
        if not isinstance(self.hidden_layers[-1], nn.Identity):
            prev_dim = self.hidden_layers[-1].out_features
        else:
            prev_dim = self.input_dim
        self.hidden_layers.append(nn.Linear(prev_dim, n_neurons).to(self.device))
        # expand output layer input dim to new n_neurons preserving weights
        self._expand_output_layer(new_in_features=n_neurons)
        return True
    
    def update_output_layer(self, new_output_dim):
        # Expand output layer to new_output_dim while preserving learned weights for existing outputs
        self._expand_output_layer(new_out_features=new_output_dim)

    
    def print_structure(self):
        print("Current MLP Structure:")
        for idx, layer in enumerate(self.hidden_layers):
            layer_type = type(layer).__name__
            if isinstance(layer, nn.Identity):
               
                print(f" Layer {idx}: Identity (no neurons)")
            else:
                
                print(f" Layer {idx}: {layer_type} with {layer.out_features} neurons")
        
        print(f" Output Layer: Linear with {self.output_layer.out_features} output classes")