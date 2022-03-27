#!/usr/bin/env python3

import trojanvision
from trojanvision.utils import superimpose
import torchvision.transforms.functional as F
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)

    if not os.path.exists('./result'):
        os.makedirs('./result')

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model)
    for data in dataset.loader['valid']:
        _input, _label, forward_kwargs = model.get_data(data)
        heatmap = model.get_heatmap(_input, _label, method='saliency_map', **forward_kwargs)
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        for i, map in enumerate(heatmap):
            F.to_pil_image(heatmap[i]).save(f'./result/heatmap_{i}.jpg')
        break
