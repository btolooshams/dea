"""
Copyright (c) 2020 Bahareh Tolooshams

data generator

:author: Bahareh Tolooshams
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np


def get_MNIST_loaders(batch_size, shuffle=False, train_batch=None, test_batch=None):
    if train_batch == None:
        train_loader = get_MNIST_loader(batch_size, trainable=True, shuffle=shuffle)
    else:
        train_loader = get_MNIST_loader(train_batch, trainable=True, shuffle=shuffle)

    if test_batch == None:
        test_loader = get_MNIST_loader(batch_size, trainable=False, shuffle=shuffle)
    else:
        test_loader = get_MNIST_loader(test_batch, trainable=False, shuffle=shuffle)
    return train_loader, test_loader


def get_MNIST_loader(batch_size, trainable=True, shuffle=False):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "../data",
            train=trainable,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


def get_VOC_loaders(
    batch_size, crop_dim=(250, 250), shuffle=False, train_batch=None, test_batch=None
):
    if train_batch == None:
        train_loader = get_VOC_loader(
            batch_size, image_set="train", crop_dim=crop_dim, shuffle=shuffle
        )
    else:
        train_loader = get_VOC_loader(
            train_batch, image_set="train", crop_dim=crop_dim, shuffle=shuffle
        )

    if test_batch == None:
        test_loader = get_VOC_loader(batch_size, image_set="val", shuffle=shuffle)
    else:
        test_loader = get_VOC_loader(test_batch, image_set="val", shuffle=shuffle)
    return train_loader, test_loader


def get_VOC_loader(batch_size, image_set, crop_dim=(250, 250), shuffle=False):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.VOCSegmentation(
            "../data",
            year="2012",
            image_set=image_set,
            download=True,
            target_transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.RandomCrop(
                        (1, 1),
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.RandomCrop(
                        crop_dim,
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


def get_VOC_loaders_detection(
    batch_size, crop_dim=(250, 250), shuffle=False, train_batch=None, test_batch=None
):
    if train_batch == None:
        train_loader = get_VOC_loader_detection(
            batch_size, image_set="train", crop_dim=crop_dim, shuffle=shuffle
        )
    else:
        train_loader = get_VOC_loader_detection(
            train_batch, image_set="train", crop_dim=crop_dim, shuffle=shuffle
        )

    if test_batch == None:
        test_loader = get_VOC_loader(batch_size, image_set="val", shuffle=shuffle)
    else:
        test_loader = get_VOC_loader(test_batch, image_set="val", shuffle=shuffle)
    return train_loader, test_loader


def get_VOC_loader_detection(batch_size, image_set, crop_dim=(250, 250), shuffle=False):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.VOCDetection(
            "../data",
            year="2012",
            image_set=image_set,
            download=True,
            target_transform=None,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.RandomCrop(
                        crop_dim,
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


def get_path_loader(batch_size, image_path, shuffle=False):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=image_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader
