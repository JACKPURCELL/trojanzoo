#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets


#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load_data.py
# Author: Yahui Liu <yahui.cvrs@gmail.com>

import cv2
import os
import numpy as np

DATA_LEN = 3072
CHANNEL_LEN = 1024
SHAPE = 32 #圖像大小

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
  im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
  print(im)
  if color == "RGB":
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = np.transpose(im, [2, 1, 0])
  if shape != None:
    assert isinstance(shape, int) 
    im = cv2.resize(im, (shape, shape))
  return im

def read_data(filename, data_path, shape=None, color='RGB'):
  """
     filename (str): a file 
       data file is stored in such format:
         image_name  label
     data_path (str): image data folder
     return (numpy): a array of image and a array of label
  """ 
  if os.path.isdir(filename):
    print( "Can't found data file!")
  else:
    f = open(filename)
    lines = f.read().splitlines()
    count = len(lines)
    data = np.zeros((count, DATA_LEN), dtype=np.uint8)
    #label = np.zeros(count, dtype=np.uint8)
    lst = [ln.split(' ')[0] for ln in lines]
    label = [int(ln.split(' ')[1]) for ln in lines]
    
    idx = 0
    s, c = SHAPE, CHANNEL_LEN
    for ln in lines:
      fname, lab = ln.split(' ')
      print(fname)
      im = imread(os.path.join(data_path, fname), shape=s, color='RGB')
      '''
      im = cv2.imread(os.path.join(data_path, fname), cv2.IMREAD_UNCHANGED)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (s, s))
      '''

      data[idx,:c] =  np.reshape(im[:,:,0], c)
      data[idx, c:2*c] = np.reshape(im[:,:,1], c)
      data[idx, 2*c:] = np.reshape(im[:,:,2], c)
      label[idx] = int(lab)
      idx = idx + 1
      
    return data, label, lst




class tea_CIFAR10(ImageSet):
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
    def get_data_full(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Process image data.
        Defaults to put input and label on ``env['device']`` with ``non_blocking``
        and transform label to ``torch.LongTensor``.

        Args:
            data (tuple[torch.Tensor, torch.Tensor]): Tuple of batched input and label.
            **kwargs: Any keyword argument (unused).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]):
                Tuple of batched input and label on ``env['device']``.
                Label is transformed to ``torch.LongTensor``.
        """
        return (data[0].to(env['device'], non_blocking=True),
                data[1].to(env['device'], dtype=torch.long, non_blocking=True),
                data[2].to(env['device'], dtype=torch.long, non_blocking=True))


class tea_CIFAR100(tea_CIFAR10):
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
