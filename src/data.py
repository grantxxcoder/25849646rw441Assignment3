import numpy as np
import pandas as pd
import torchvision
from sklearn.datasets import fetch_covtype

def load_covtype_data():
    data_cov = fetch_covtype(as_frame=True)
    df_covtype = data_cov.frame
    df_covtype['Cover_Type'] = df_covtype['Cover_Type'] - 1
    df_covtype['Aspect_sin'] = np.sin(np.deg2rad(df_covtype['Aspect']))
    df_covtype['Aspect_cos'] = np.cos(np.deg2rad(df_covtype['Aspect']))
    df_covtype = df_covtype.drop(columns=['Aspect'])
    return df_covtype

def load_fashion_mnist_data():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_images = [img.numpy().flatten() for img, _ in train_dataset]
    train_labels = [label for _, label in train_dataset]
    df_train_fashion = pd.DataFrame(train_images)
    df_train_fashion['label'] = train_labels

    test_images = [img.numpy().flatten() for img, _ in test_dataset]
    test_labels = [label for _, label in test_dataset]
    df_test_fashion = pd.DataFrame(test_images)
    df_test_fashion['label'] = test_labels

    return df_train_fashion, df_test_fashion