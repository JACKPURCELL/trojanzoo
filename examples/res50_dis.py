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
    
    #init resnet50 model
    # kwargs['official'] = True
    # kwargs['model_name'] = 'resnet50_comp'
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
    model._distillation(**trainer)



