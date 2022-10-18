#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0,1,2 python examples/distillation.py --verbose 1 --dataset cifar10 --model stu_nats_bench --model_index 1001 --model_seed 777 --batch_size 512 --lr 0.025 --lr_scheduler --lr_min 1e-3 --grad_clip 5.0 --epochs 200 --download --nats_path /data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full --adv_train free --validate_interval 1 --save --official --color 
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
    # config = {
    #     'name': 'DARTS-V1',
    #     'C': 16,
    #     'N': 5,
    #     'max_nodes': 4,
    #     'num_classes': model.num_classes,
    #     'space': [
    #         "none",
    #         "skip_connect",
    #         "nor_conv_1x1",
    #         "nor_conv_3x3",
    #         "avg_pool_3x3",
    #     ],
    #     'affine': True,
    #     'track_running_stats': True,
    # }
    
    
    # network = model.get_cell_based_tiny_net(config)
    # model._model.load_model(network)
    # model._model.to(env['device'])
    
    kwargs['model_name'] = 'nats_bench'
    

    # kwargs['model'] = 'DARTS'

    tea_model = trojanvision.models.create(dataset=dataset, **kwargs)
    # filename = "/root/work/trojanzoo/cifar10_model.pt"
    # filename = "/home/jkl6486/trojanzoo/cifar10_model.pt"
    
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
    def fun(variable):
        num = ['0','1','2','3','4','5','6','7','8','9','']
        if (variable in num):
            return False
        else:
            return True
    tea_arch_list = list(filter(fun, re.split('\+|\||~',tea_model.arch_str)))
    # ['nor_conv_1x1', 'none', 'nor_conv_1x1', 'skip_connect', 'skip_connect', 'nor_conv_3x3']
    op_list =  [
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    tea_arch_tensor = torch.zeros(6, 5).cuda()
    for i in range(0, len(tea_arch_list)):
        index=op_list.index(tea_arch_list[i])
        tea_arch_tensor[i][index] = 1
        

    
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    # filename = "./data/model/image/cifar10/stu_nats_bench_at-free.pth"

    # model.load(filename)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
        trojanvision.summary(env=env, dataset=dataset, model=tea_model)
    print("=====================TEACHER VALIDATE==================")
    # acc, loss = tea_model._validate()
    print("===================Start training================")
    model._distillation(tea_forward_fn=tea_model.__call__, tea_arch_tensor=tea_arch_tensor, tea_arch_list=tea_arch_list, **trainer)

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
