import pytest
from src.data import load_covtype_data, load_fashion_mnist_data

def test_load_covtype_data():
    df = load_covtype_data()
    assert df is not None
    assert 'Cover_Type' in df.columns
    assert df.shape[0] > 0

def test_load_fashion_mnist_data():
    train_df, test_df = load_fashion_mnist_data()
    assert train_df is not None
    assert test_df is not None
    assert 'label' in train_df.columns
    assert 'label' in test_df.columns
    assert train_df.shape[0] > 0
    assert test_df.shape[0] > 0