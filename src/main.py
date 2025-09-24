import os
import torch
from src.config import DATA_DIR, RESULTS_DIR
from src.data import load_covtype_data, load_fashion_mnist_data
from src.preprocessing import preprocess_covtype_data, preprocess_fashion_data
from src.models.simple_mlp import SimpleMLP
from src.models.dynamic_mlp import DynamicMLP
from src.training import train_model, train_complex_model
from src.evaluation import evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load and preprocess Covtype data
    covtype_data = load_covtype_data()
    X_cov, y_cov = preprocess_covtype_data(covtype_data)

    # Load and preprocess Fashion-MNIST data
    fashion_data = load_fashion_mnist_data()
    X_fashion, y_fashion = preprocess_fashion_data(fashion_data)

    # Initialize models
    simple_mlp = SimpleMLP(input_dim=X_cov.shape[1], output_dim=len(set(y_cov))).to(device)
    dynamic_mlp_cov = DynamicMLP(input_dim=X_cov.shape[1]).to(device)
    # dynamic_mlp = DynamicMLP(input_dim=X_fashion.shape[1]).to(device)

    # Train models
    train_complex_model(dynamic_mlp_cov, X_cov, y_cov)
    # train_model(dynamic_mlp, X_fashion, y_fashion)

    # Evaluate models
    evaluate_model(dynamic_mlp_cov, X_cov, y_cov)
    # evaluate_model(dynamic_mlp, X_fashion, y_fashion)

if __name__ == "__main__":
    main()