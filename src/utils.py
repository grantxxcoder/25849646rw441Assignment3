from matplotlib import pyplot as plt
import pandas as pd

def plot_learning_curves(train_acc, test_acc, train_losses, val_losses, title="Learning Curves"):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

def save_results_to_csv(results, filepath):
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

def load_results_from_csv(filepath):
    return pd.read_csv(filepath)