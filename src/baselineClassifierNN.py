import torch.nn as nn
import torch
import numpy as np
import random

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation='relu', seed=42):
        super(SimpleMLP, self).__init__()
        random.seed(seed)  # Set seed for Python's random module
        np.random.seed(seed)  # Set seed for NumPy
        torch.manual_seed(seed)  # Set seed for PyTorch (CPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # Set seed for CUDA (single GPU)
            torch.cuda.manual_seed_all(seed)  # Set seed for CUDA (all GPUs)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False
        
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