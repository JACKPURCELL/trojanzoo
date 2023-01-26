# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
import torchvision.datasets as datasets
from trojanvision.datasets.imagefolder import ImageFolder
from trojanvision.environ import env
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
import re
from trojanvision.utils.dataset import ZipFolder
import hapi
# hapi.config.data_dir = "/home/ljc/HAPI" 

class _EXPW(datasets.ImageFolder):
    
    def __init__(self, mode:str=None, hapi_data_dir:str = None, hapi_info:str = None, **kwargs):

        super().__init__(**kwargs)
        hapi.config.data_dir = hapi_data_dir
        # hapi.download()
        if mode is None:
            mode = self.root.split('/')[-1]

        dic = hapi_info 

        dic_split = dic.split('/')
        predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])

        self.info_lb = torch.zeros(40000,dtype=torch.long)
        self.info_conf = torch.zeros(40000)

        
        for i in range(len(predictions[dic])):
            hapi_id = int(predictions[dic][i]['example_id'].split('.')[0])
            self.info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
            self.info_conf[hapi_id] = torch.tensor((predictions[dic][i]['confidence']))

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        hapi_id = torch.tensor(int(path.split('/')[-1].split('.')[0]))
        hapi_label = self.info_lb[hapi_id]
        hapi_confidence = self.info_conf[hapi_id]
        other_confidence = (1 - hapi_confidence)/6
        soft_label = torch.ones(7)*other_confidence
        soft_label[int(hapi_label)] = hapi_confidence
        # gt_label = self.info_gt[hapi_id][0
        
        return sample, target, soft_label, hapi_label
    
class EXPW_112(ImageFolder):
    name = 'EXPW_112'
    num_classes = 7
    data_shape = [3,112,112]

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        r"""Add image dataset arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete dataset class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.

        See Also:
            :meth:`trojanvision.datasets.ImageSet.add_argument()`
        """
        super().add_argument(group)
        group.add_argument('--hapi_data_dir', 
                           help='hapi_data_dir')
        group.add_argument('--hapi_info', 
                           help='hapi_info')
        return group
    
    def __init__(self, hapi_data_dir:str = None, hapi_info:str = None, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225]}, **kwargs):
        self.hapi_data_dir = hapi_data_dir
        self.hapi_info = hapi_info
        super().__init__(norm_par=norm_par, **kwargs)

    def _get_org_dataset(self, mode: str, data_format: str = None,
                         **kwargs) -> datasets.DatasetFolder:
        data_format = data_format or self.data_format
        root = os.path.join(self.folder_path, mode)
        DatasetClass = _EXPW
        if data_format == 'zip':
            root = os.path.join(self.folder_path,
                                f'{self.name}_{mode}_store.zip')
            DatasetClass = ZipFolder
            if 'memory' not in kwargs.keys():
                kwargs['memory'] = self.memory
        return DatasetClass(root=root, hapi_data_dir=self.hapi_data_dir, hapi_info=self.hapi_info, mode=mode, **kwargs)
    
    def get_data(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                data[2].to(env['device'], non_blocking=True),
                data[3].to(env['device'], dtype=torch.long, non_blocking=True))

 
class EXPW_224(ImageFolder):
    name = 'EXPW_224'
    num_classes = 7

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        r"""Add image dataset arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete dataset class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.

        See Also:
            :meth:`trojanvision.datasets.ImageSet.add_argument()`
        """
        super().add_argument(group)
        group.add_argument('--hapi_data_dir', 
                           help='hapi_data_dir')
        group.add_argument('--hapi_info', 
                           help='hapi_info')
        return group
    
    def __init__(self, hapi_data_dir:str = None, hapi_info:str = None, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225]}, **kwargs):
        self.hapi_data_dir = hapi_data_dir
        self.hapi_info = hapi_info
        super().__init__(norm_par=norm_par, **kwargs)

    def _get_org_dataset(self, mode: str, data_format: str = None,
                         **kwargs) -> datasets.DatasetFolder:
        data_format = data_format or self.data_format
        root = os.path.join(self.folder_path, mode)
        DatasetClass = _EXPW
        if data_format == 'zip':
            root = os.path.join(self.folder_path,
                                f'{self.name}_{mode}_store.zip')
            DatasetClass = ZipFolder
            if 'memory' not in kwargs.keys():
                kwargs['memory'] = self.memory
        return DatasetClass(root=root, hapi_data_dir=self.hapi_data_dir, hapi_info=self.hapi_info, mode=mode, **kwargs)
    
    def get_data(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                data[2].to(env['device'], non_blocking=True),
                data[3].to(env['device'], dtype=torch.long, non_blocking=True))
