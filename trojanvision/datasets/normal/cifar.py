#!/usr/bin/env python3

import random

import torch
from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

# class _CIFAR10(datasets.CIFAR10):
#     """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

#     Args:
#         root (string): Root directory of dataset where directory
#             ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
#         train (bool, optional): If True, creates dataset from training set, otherwise
#             creates from test set.
#         transform (callable, optional): A function/transform that takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.

#     """

#     base_folder = "cifar-10-batches-py"
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     filename = "cifar-10-python.tar.gz"
#     tgz_md5 = "c58f30108f718f92721af3b95e74349a"
#     train_list = [
#         ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
#         ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
#         ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
#         ["data_batch_4", "634d18415352ddfa80567beed471001a"],
#         ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
#     ]

#     test_list = [
#         ["test_batch", "40351d587109b95175f43aff81a1287e"],
#     ]
#     meta = {
#         "filename": "batches.meta",
#         "key": "label_names",
#         "md5": "5ff9c542aee3614f3951f8cda6e48888",
#     }

#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:

#         super().__init__(root, transform=transform, target_transform=target_transform)

#         self.train = train  # training set or test set

#         if download:
#             self.download()

#         if not self._check_integrity():
#             raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

#         if self.train:
#             downloaded_list = self.train_list
#         else:
#             downloaded_list = self.test_list

#         self.data: Any = []
#         self.targets = []

#         # now load the picked numpy arrays
#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, "rb") as f:
#                 entry = pickle.load(f, encoding="latin1")
#                 self.data.append(entry["data"])
#                 if "labels" in entry:
#                     self.targets.extend(entry["labels"])
#                 else:
#                     self.targets.extend(entry["fine_labels"])

#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

#         self._load_meta()

#     def _load_meta(self) -> None:
#         path = os.path.join(self.root, self.base_folder, self.meta["filename"])
#         if not check_integrity(path, self.meta["md5"]):
#             raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
#         with open(path, "rb") as infile:
#             data = pickle.load(infile, encoding="latin1")
#             self.classes = data[self.meta["key"]]
#         self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         if self.train:

#             # Choose another image/label randomly
#             mixup_idx = random.randint(0, len(self.data)-1)
#             mixup_label = torch.zeros(10)
#             target[self.targets[mixup_idx]] = 1.
#             if self.transform:
#                 mixup_image = self.transform(self.data[mixup_idx])

#             # Select a random number from the given beta distribution
#             # Mixup the images accordingly
#             alpha = 0.2
#             lam = np.random.beta(alpha, alpha)
#             image = lam * image + (1 - lam) * mixup_image
#             label = lam * label + (1 - lam) * mixup_label
            
#         return img, target

#     def __len__(self) -> int:
#         return len(self.data)

#     def _check_integrity(self) -> bool:
#         root = self.root
#         for fentry in self.train_list + self.test_list:
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True

#     def download(self) -> None:
#         if self._check_integrity():
#             print("Files already downloaded and verified")
#             return
#         download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

#     def extra_repr(self) -> str:
#         split = "Train" if self.train is True else "Test"
#         return f"Split: {split}"

class CIFAR10(ImageSet):
    r"""CIFAR10 dataset introduced by Alex Krizhevsky in 2009.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.CIFAR10`
        * paper: `Learning Multiple Layers of Features from Tiny Images`_
        * website: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes:
        name (str): ``'cifar10'``
        num_classes (int): ``10``
        data_shape (list[int]): ``[3, 32, 32]``
        class_names (list[str]):
            | ``['airplane', 'automobile', 'bird', 'cat', 'deer',``
            | ``'dog', 'frog', 'horse', 'ship', 'truck']``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.49139968, 0.48215827, 0.44653124],``
            | ``'std'  : [0.24703233, 0.24348505, 0.26158768]}``

    .. _Learning Multiple Layers of Features from Tiny Images:
        https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.49139968, 0.48215827, 0.44653124],
                                                           'std': [0.24703233, 0.24348505, 0.26158768], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.CIFAR10(root=self.folder_path, train=True, download=True)
        datasets.CIFAR10(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs) -> datasets.CIFAR10:
        assert mode in ['train', 'valid']
        return datasets.CIFAR10(root=self.folder_path, train=(mode == 'train'), **kwargs)


class CIFAR100(CIFAR10):
    r"""CIFAR100 dataset. It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.CIFAR100`
        * paper: `Learning Multiple Layers of Features from Tiny Images`_
        * website: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes:
        name (str): ``'cifar100'``
        num_classes (int): ``100``
        data_shape (list[int]): ``[3, 32, 32]``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.49139968, 0.48215827, 0.44653124],``
            | ``'std'  : [0.24703233, 0.24348505, 0.26158768]}``

    .. _Learning Multiple Layers of Features from Tiny Images:
        https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    name = 'cifar100'
    num_classes = 100

    def initialize(self):
        datasets.CIFAR100(root=self.folder_path, train=True, download=True)
        datasets.CIFAR100(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs) -> datasets.CIFAR100:
        assert mode in ['train', 'valid']
        return datasets.CIFAR100(root=self.folder_path, train=(mode == 'train'), **kwargs)
