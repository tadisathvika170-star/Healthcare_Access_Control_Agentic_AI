"""
dataset.py

Creates PyTorch DataLoaders for training
LSTM and GRU models.
"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from dataset.preprocess import DataPreprocessor


class KeystrokeDataset:
    """
    Loads the processed dataset and creates
    PyTorch DataLoaders.
    """

    def __init__(self, batch_size=32):

        self.batch_size = batch_size

        processor = DataPreprocessor()

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = processor.preprocess()

        # ---------------------------------
        # Convert to Torch Tensors
        # ---------------------------------

        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)

        self.y_train = torch.LongTensor(y_train.to_numpy(copy=True))
        self.y_test = torch.LongTensor(y_test.to_numpy(copy=True))
        # ---------------------------------
        # Reshape for LSTM / GRU
        # (batch_size, sequence_length, features)
        # ---------------------------------

        self.X_train = self.X_train.unsqueeze(1)
        self.X_test = self.X_test.unsqueeze(1)

    def train_loader(self):

        dataset = TensorDataset(
            self.X_train,
            self.y_train,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_loader(self):

        dataset = TensorDataset(
            self.X_test,
            self.y_test,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":

    dataset = KeystrokeDataset()

    train_loader = dataset.train_loader()
    test_loader = dataset.test_loader()

    print("Training batches :", len(train_loader))
    print("Testing batches  :", len(test_loader))

    X, y = next(iter(train_loader))

    print()
    print("Input Shape :", X.shape)
    print("Output Shape:", y.shape)