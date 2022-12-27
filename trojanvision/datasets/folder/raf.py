# -*- coding: utf-8 -*-
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
hapi.config.data_dir = "/home/jkl6486/HAPI" 
hapi.download()
class _RAF(datasets.ImageFolder):
    
    def __init__(self, mode:str=None, **kwargs):

        super().__init__(**kwargs)
        mode = self.root.split('/')[-1]
        if mode == 'valid':
            mode = 'test'
        predictions =  hapi.get_predictions(task="fer", dataset="rafdb", date="22-05-23", api=["google_fer"])
        self.info = torch.zeros(len(self.targets) + 1,2)
        dic = str('fer/rafdb/google_fer/22-05-23')
        for i in range(len(predictions[dic])):
            hapi_mode = predictions[dic][i]['example_id'].split('_')[0]
            hapi_id = int(predictions[dic][i]['example_id'].split('_')[1])
            if hapi_mode == mode:
                self.info[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label'], predictions[dic][i]['confidence']))

        # self.idx2name = list_string(numpy)/torch.tensor (zip取0)
    # def __getitem__():
    #     idx2name[i]
    #     return image, idx2name[i] ,label
    
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

        hapi_id = torch.tensor(int(re.findall(r'_(.*).jpg', path)[0]))
        hapi_label = self.info[hapi_id][0]
        hapi_confidence = self.info[hapi_id][1]
        other_confidence = (1 - hapi_confidence)/6
        soft_label = torch.ones(7)*other_confidence
        soft_label[int(hapi_label)] = hapi_confidence
        
        return sample, target, soft_label, hapi_label
    
class RAF(ImageFolder):
    name = 'RAFDB'
    num_classes = 7
    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225]}, **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)
    
    def _get_org_dataset(self, mode: str, data_format: str = None,
                         **kwargs) -> datasets.DatasetFolder:
        data_format = data_format or self.data_format
        root = os.path.join(self.folder_path, mode)
        DatasetClass = _RAF
        if data_format == 'zip':
            root = os.path.join(self.folder_path,
                                f'{self.name}_{mode}_store.zip')
            DatasetClass = ZipFolder
            if 'memory' not in kwargs.keys():
                kwargs['memory'] = self.memory
        return DatasetClass(root=root, **kwargs)
    
    def get_data(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
                data[2].to(env['device'], non_blocking=True),
                data[3].to(env['device'], dtype=torch.long, non_blocking=True))
        
    # def get_data(data: tuple[torch.Tensor, torch.Tensor], adv_train: bool = False,
    #                 mode: str = 'train_STU', **kwargs) -> tuple[torch.Tensor, torch.Tensor]:

    #     if mode == 'train_STU' or mode == 'train_ADV_STU':
    #         _input, _label = get_data_old(data, adv_train=adv_train, **kwargs)
    #         _soft_label = tea_forward_fn(_input,**kwargs)
    #         _soft_label.detach()
    #         # _output = self(_input, **kwargs)
    #         return _input, _label, _soft_label
    #     elif mode =='valid':
    #         _input, _label = get_data_old(data, adv_train=adv_train, **kwargs)
    #         return _input, _label
                        
    #     def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
    #              adv_train: bool = False,
    #              **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    #     r"""
    #     Args:
    #         data (tuple[torch.Tensor, torch.Tensor]):
    #             Tuple of ``(input, label)``.
    #         adv_train (bool): Whether to Defaults to ``False``.
    #         **kwargs: Keyword arguments passed to
    #             :meth:`trojanzoo.models.Model.get_data()`.
    #     """
    #     # In training process, `adv_train` args will not be passed to `get_data`.
    #     # It's passed to `_validate`.
    #     # So it's always `False`.
    #     _input, _label = super().get_data(data, **kwargs)
    #     if adv_train:
    #         assert self.pgd is not None
    #         adv_x, _ = self.pgd.optimize(_input=_input, target=_label)
    #         return adv_x, _label
    #     return _input, _label
    
    
    # def get_data:
    #     imgs_name = find(index)in dataset(_xxx)
        
    # def __getitem__(self, index: int | slice):
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #     return img, target

    # def __len__(self):
    #     return len(self.file_paths)

    # def __getitem__(self, idx):
    #     label = self.label[idx]
    #     image = cv2.imread(self.file_paths[idx])
            
    #     image = image[:, :, ::-1]


    #     return image, label, idx
    
    
# class RafDataset(ImageFolder):
#     name = 'RAFDB'
#     num_classes = 7
#     # data_shape = [3, 100, 100]
    
    
#     def __init__(self, mode: str = 'train', norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
#                                                            'std': [0.229, 0.224, 0.225]}, **kwargs):
#         data_dir = '/home/jkl6486/Erasing-Attention-Consistency/raf-basic/'
#         self.mode = mode
#         df = pd.read_csv(os.path.join(data_dir, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
#         name_c = 0
#         label_c = 1
#         if mode == 'train':
#             dataset = df[df[name_c].str.startswith('train')]
#         else:
#             dataset = df[df[name_c].str.startswith('test')]
            
#         self.label = dataset.iloc[:, label_c].values - 1
#         images_names = dataset.iloc[:, name_c].values

#         self.file_paths = []
#         # self.clean = (args.label_path == 'list_patition_label.txt')
#         self.clean = True
        
#         for f in images_names:
#             f = f.split(".")[0]
#             f += '.jpg'
#             file_name = os.path.join(data_dir, f)
#             self.file_paths.append(file_name)

#         self.norm_par = norm_par
#         self.normalize = False

        
#         # super().__init__(norm_par=norm_par, **kwargs)

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         label = self.label[idx]
#         image = cv2.imread(self.file_paths[idx])
            
#         image = image[:, :, ::-1]


#         return image, label, idx
# def add_g(image_array, mean=0.0, var=30):
#     std = var ** 0.5
#     image_add = image_array + np.random.normal(mean, std, image_array.shape)
#     image_add = np.clip(image_add, 0, 255).astype(np.uint8)
#     return image_add

# def flip_image(image_array):
#     return cv2.flip(image_array, 1)

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
    
# def generate_flip_grid(w, h, device):
#     # used to flip attention maps
#     x_ = torch.arange(w).view(1, -1).expand(h, -1)
#     y_ = torch.arange(h).view(-1, 1).expand(-1, w)
#     grid = torch.stack([x_, y_], dim=0).float().to(device)
#     grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
#     grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
#     grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
#     grid[:, 0, :, :] = -grid[:, 0, :, :]
#     return grid

# class RafDataset(ImageSet):
#     name = 'RAFDB'
#     num_classes = 10
#     data_shape = [3, 100, 100]
    
    
#     def __init__(self, mode: str = 'train', basic_aug=True, transform=None, **kwargs):


#         self.mode = mode
#         self.basic_aug = basic_aug
#         self.transform = transform
#         df = pd.read_csv(os.path.join(root_dir, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)

        
#         name_c = 0
#         label_c = 1
#         if mode == 'train':
#             dataset = df[df[name_c].str.startswith('train')]
#         else:
#             dataset = df[df[name_c].str.startswith('test')]
            
#         self.label = dataset.iloc[:, label_c].values - 1
#         images_names = dataset.iloc[:, name_c].values
#         self.aug_func = [flip_image, add_g]
#         self.file_paths = []
#         # self.clean = (args.label_path == 'list_patition_label.txt')
#         self.clean = True
        
#         for f in images_names:
#             f = f.split(".")[0]
#             f += '_aligned.jpg'
#             file_name = os.path.join(self.raf_path, 'Image/aligned', f)
#             self.file_paths.append(file_name)


#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         label = self.label[idx]
#         image = cv2.imread(self.file_paths[idx])
            
#         image = image[:, :, ::-1]
        

#         if self.phase == 'train':
#             if self.basic_aug and random.uniform(0, 1) > 0.5:
#                 image = self.aug_func[1](image)

#         if self.transform is not None:
#             image = self.transform(image)
      

#         return image, label, idx