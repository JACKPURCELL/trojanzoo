# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms

from trojanvision.datasets.imagefolder import ImageFolder
from trojanvision.environ import env

class RafDataset(ImageFolder):
    name = 'RAFDB'
    num_classes = 7

    
    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225]}, **kwargs):

        super().__init__(norm_par=norm_par, **kwargs)

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