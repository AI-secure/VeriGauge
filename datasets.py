from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
# os.environ[IMAGENET_LOC_ENV] = "/data/datasets/imagenet/ILSVRC2012"
os.environ[IMAGENET_LOC_ENV] = "/srv/local/data/ImageNet/ILSVRC2012_full"

# list of all datasets
from constants import DATASETS


def get_dataset(dataset: str, split: str, normalize=None) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split, normalize)
    elif dataset == "cifar10":
        return _cifar10(split, normalize)
    elif dataset == "mnist":
        return _mnist(split, normalize)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10


def get_input_shape(dataset: str):
    """Return a list of integer indicating the input shape as (num_channel, height, weight)"""
    if dataset == "imagenet":
        return (3, 224, 224)
    elif dataset == 'cifar10':
        return (3, 32, 32)
    elif dataset == 'mnist':
        return (1, 28, 28)


def _mnist(split: str, normalize) -> Dataset:
    if normalize is None:
        transform = transforms.ToTensor()
    else:
        mean, std = normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transform)
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transform)


def _cifar10(split: str, normalize) -> Dataset:
    transform_list = list()
    if split == "train":
        transform_list.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    transform_list.append(transforms.ToTensor())
    if normalize is not None:
        mean, std = normalize
        transform_list.extend([transforms.Normalize(mean=mean, std=std)])
    transform = transforms.Compose(transform_list)

    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transform)
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transform)


def _imagenet(split: str, normalize) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]

    transform_list = list()
    if split == "train":
        transform_list.extend([transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip()])
    elif split == "test":
        transform_list.extend([transforms.Scale(256), transforms.CenterCrop(224)])
    transform_list.append(transforms.ToTensor())
    if normalize is not None:
        transform_list.extend(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_list)

    if split == "train":
        subdir = os.path.join(dir, "train")
    elif split == "test":
        subdir = os.path.join(dir, "val")
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.orig_means = means
        self.orig_sds = sds
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
