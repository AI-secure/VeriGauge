import torch
from torch import nn
from torchvision.transforms import transforms
from torchvision import datasets


def data_loaders(dataset, batch_size, shuffle_test=False, norm_mean=None, norm_std=None):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    if dataset == 'mnist':
        train = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   normalize,
                               ])
        )
        test = datasets.MNIST('./data', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  normalize,
                              ])
        )
    elif dataset == 'cifar':
        train = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      normalize,
                                  ]))
        test = datasets.CIFAR10('./data', train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader
