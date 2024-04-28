# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import numpy as np
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PNDataset(Dataset):
    """
    create a dataset for PointNet
    Note that the input data should be (samples, points, coordinates)
    while the output data should be (samples, coordinates, points) due to conv1d

    """
    def __init__(
            self, X:np.array, y:np.array=None, num_points:int=None, transform=None
            ):
        """
        Parameters
        ----------
        X: np.array
            an array (samples, points, coordinates)

        y: np.array
            an array (samples, labels)

        """
        if input is None:
            raise ValueError('!! Give input !!')
        if type(transform) == list:
            if len(transform) != 0:
                if transform[0] is None:
                    self.transform = []
                else:
                    self.transform = transform
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = []
            else:
                self.transform = [transform]
        self.X = X
        self.y = y # labels
        self.datanum = len(X)
        self.num_points = num_points
        self._indices = list(range(X.shape[1])) # shuffle用


    def __len__(self):
        return self.datanum


    def __getitem__(self, idx):
        data = self.X[idx]
        i = self._indices.copy()
        np.random.shuffle(i)
        data = data[i][:self.num_points].T # conv1dを使うので(coordinates, points)へ
        data = torch.from_numpy(data.astype(np.float32))
        if len(self.transform) > 0:
            for t in self.transform:
                data = t(data)
        if self.y is None:
            return data, None
        label = self.y[idx]
        label = torch.from_numpy(label.astype(np.uint8))
        return data, label


class AddNoise:
    def __init__(self, ratio:float=1e-2) -> None:
        """
        Parameters
        ----------
        ratio: float
            ratio of augumentation
            add white noise to the input data
            whose mean is 0 and std is mode(data) * ratio

        """
        self.ratio = ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x)
        noise = torch.normal(0, std * self.ratio, size=x.size())
        x = x + noise
        return x


def prep_dataset(X, y, num_points, transform=None) -> Dataset:
    """
    prepare dataset from row data

    Parameters
    ----------
    data: array
        input data such as np.array
        (n_samples, n_features)

    transform: a list of transform functions
        each function should return torch.tensor by __call__ method

    """
    mydataset = PNDataset(X, y, num_points, transform)
    return mydataset


def generate_subset(
        x:np.array, y:np.array=None, ratio:float=0.1, random_seed:int=0
        ):
    """
    generate a subset of the dataset

    Parameters
    ----------
    x: np.array
        input data such as np.array
        (n_samples, n_features)

    ratio: float
        the ratio of the subset to the original dataset

    random_seed: int
        the random seed for reproducibility

    """
    # set seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # generate sizes
    n_samples = len(x)
    size_test = int(n_samples * ratio)
    size_train = n_samples - size_test
    # shuffle and split
    indices = np.random.permutation(n_samples)
    x_train = x[indices[:size_train]]
    x_test = x[indices[size_train:]]
    if y is None:
        return x_train, x_test, None, None
    y_train = y[indices[:size_train]]
    y_test = y[indices[size_train:]]
    return x_train, x_test, y_train, y_test


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> DataLoader:
    """
    prepare train and test loader

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance

    batch_size: int
        the batch size

    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing

    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing

    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )
    return loader


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_data(
    x_train:np.array, x_test:np.array=None, y_train:np.array=None, y_test:np.array=None,
    num_points:int=256, batch_size:int=32,
    transform=(None, None), shuffle=(True, False),
    num_workers=2, pin_memory=True, ratio:float=0.01,
    random_seed=0
    ) -> Tuple[DataLoader, DataLoader]:
    """
    prepare train and test loader from data
    combination of prep_dataset and prep_dataloader for model building

    Parameters
    ----------
    x_train, x_test: np.array
        training and test data (n_samples, n_features)

    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list
        default: (None, None)

    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing

    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing

    """
    if x_test is None:
        x_train, x_test, y_train, y_test = generate_subset(
            x_train, y_train, ratio, random_seed
            )
    train_tf = transform[0]
    if train_tf is None:
        train_tf = [AddNoise(ratio)]
    train_dataset = prep_dataset(x_train, y_train, num_points, train_tf)
    test_dataset = prep_dataset(x_test, y_test, num_points, transform[1])
    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )
    return train_loader, test_loader
