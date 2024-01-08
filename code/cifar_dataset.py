# This code is a copy of https://raw.githubusercontent.com/passalis/pkth/master/loaders/cifar_dataset.py
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def cifar10_loader(
    data_path="../data",
    batch_size=128,
    maxsize=-1,
):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_data = dset.CIFAR10(data_path, train=True, transform=transform, download=True)
    if maxsize > 0:
        train_data = torch.utils.data.Subset(train_data, list(range(maxsize)))

    test_data = dset.CIFAR10(data_path, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    return train_loader, test_loader
