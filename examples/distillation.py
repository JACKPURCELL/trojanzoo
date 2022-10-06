#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=2 python examples/distillation.py --color --validate_interval 1 --verbose 1 --dataset cifar10 --model tea_darts --supernet --arch_search --layers 8 --init_channels 16 --batch_size 80 --lr 0.025 --lr_scheduler --lr_min 1e-3 --grad_clip 5.0 --epochs 100 --save --download
"""  # noqa: E501

import re
import trojanvision
import argparse
import torch
from time import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    # trojanvision.attacks.add_argument(parser)
    torch.random.manual_seed(int(time()))
    
    kwargs = parser.parse_args().__dict__


    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    config = {
        'name': 'DARTS-V1',
        'C': 16,
        'N': 5,
        'max_nodes': 4,
        'num_classes': model.num_classes,
        'space': [
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        ],
        'affine': True,
        'track_running_stats': True,
    }
    
    
    network = model.get_cell_based_tiny_net(config)
    model._model.load_model(network)
    model._model.to(env['device'])
    
    kwargs['model_name'] = 'nats_bench'
    

    # kwargs['model'] = 'DARTS'

    tea_model = trojanvision.models.create(dataset=dataset, **kwargs)
    # filename = "/root/work/trojanzoo/cifar10_model.pt"
    # filename = "/home/jkl6486/trojanzoo/cifar10_model.pt"
    # filename = "./data/model/image/cifar10/darts_supernet_150ep.pth"
    
    # tea_model.load(filename)
    
    print("=====================AFTER LOAD TEACHER==================")
    # print(tea_model.genotype)
    print("=====================AFTER LOAD TEACHER==================")
    # stu_arch_list = model._model.features.cells.alphas
    # tea_arch_list = torch.zeros_like(stu_arch_list)
    # stu_features = model._model.features.cells
    # tea_features = tea_model._model.features.cells
    
    # for i in range(0, len(stu_features)):
    #     for j in range(0,len(stu_features[0])):
    #         if tea_features[i]==stu_features[i][j]:
    #             tea_arch_list[i][j]=1
    #             break
                
    # stu_features[0].edges['3<-2'][0]
    tea_arch_list = list(filter(None, re.split('\+|\|',tea_model.arch_str)))
   
    
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
        trojanvision.summary(env=env, dataset=dataset, model=tea_model)
    print("=====================TEACHER VALIDATE==================")
    acc, loss = tea_model._validate()
    print("===================Start training================")
    model._distillation(tea_forward_fn=tea_model.__call__, tea_arch_list=tea_arch_list, **trainer)

    # kwargs['model_name'] = 'tea_darts'


    #....................
    # Used for python interactive window

    # parser = argparse.ArgumentParser()
    # trojanvision.environ.add_argument(parser)
    # trojanvision.datasets.add_argument(parser)
    # trojanvision.models.add_argument(parser)
    # kwargs = parser.parse_args().__dict__

    # env = trojanvision.environ.create(**kwargs)
    # dataset = trojanvision.datasets.create(**kwargs)
    # model = trojanvision.models.create(dataset=dataset, **kwargs)

    # if env['verbose']:
    #     trojanvision.summary(env=env, dataset=dataset, model=model)
    # acc, loss = model._validate()
