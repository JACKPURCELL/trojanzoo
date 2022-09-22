#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=2 python examples/distillation.py --color --validate_interval 1 --verbose 1 --dataset cifar10 --model tea_darts --supernet --arch_search --layers 8 --init_channels 16 --batch_size 80 --lr 0.025 --lr_scheduler --lr_min 1e-3 --grad_clip 5.0 --epochs 100 --save --download
"""  # noqa: E501

import trojanvision
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    kwargs = parser.parse_args().__dict__


    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    kwargs['model_name'] = 'darts'
    # kwargs['model'] = 'DARTS'

    tea_model = trojanvision.models.create(dataset=dataset, **kwargs)
    # filename = "/root/work/trojanzoo/cifar10_model.pt"
    # filename = "/home/jkl6486/trojanzoo/cifar10_model.pt"
    filename = "./data/model/image/cifar10/darts_supernet_150ep.pth"
    
    tea_model.load(filename)
    tea_arch_parameters = tea_model.arch_parameters()

    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
        trojanvision.summary(env=env, dataset=dataset, model=tea_model)
    acc, loss = tea_model._validate()
    print("===================Start training================")
    model._distillation(tea_forward_fn=tea_model.__call__, tea_arch_parameters=tea_arch_parameters, **trainer)

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
