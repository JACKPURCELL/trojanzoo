#!/usr/bin/env python3

r"""--nats_path /data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full"""  # noqa: E501

import itertools
import re
from typing import Generator, Iterator, Mapping, TYPE_CHECKING

from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import _ImageModel, ImageModel
import torch.nn.functional as F
from trojanzoo.utils.fim import KFAC, EKFAC

from trojanzoo.utils.model import ExponentialMovingAverage
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
from collections import OrderedDict

import argparse
from typing import Any
from collections.abc import Callable

if TYPE_CHECKING:
    import torch.utils.data

class DARTSCells(nn.ModuleList):
    def __init__(self, cells: nn.ModuleList, alphas: nn.Parameter):
        super().__init__(cells)
        self.alphas = alphas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alphas = self.alphas.softmax(dim=-1)
        for cell in self:
            if 'search' in cell.__class__.__name__.lower():
                x = cell(x, alphas)
            else:
                x = cell(x)
        return x

    def arch_str(self) -> str:
        genotypes = []
        for i in range(1, self[0].max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.alphas[self[0].edge2index[node_str]]
                    op_name = self[0].op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        from xautodl.models.cell_searchs.genotypes import Structure   # type: ignore
        return Structure(genotypes).tostr()


class _NATSbench(_ImageModel):

    def __init__(self, network: nn.Module = None, **kwargs):
        super().__init__(**kwargs)
        self.load_model(network)

    def load_model(self, network: nn.Module):
        if 'darts' in network.__class__.__name__.lower():
            self.features = nn.Sequential(OrderedDict([
                ('stem', getattr(network, 'stem')),
                ('cells', DARTSCells(network.cells, network.arch_parameters)),
                ('lastact', getattr(network, 'lastact')),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('stem', getattr(network, 'stem')),
                ('cells', nn.Sequential(*getattr(network, 'cells'))),
                ('lastact', getattr(network, 'lastact')),
            ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', getattr(network, 'classifier'))
        ]))

    def arch_parameters(self) -> list[torch.Tensor]:
        return [self.features.cells.alphas]

    def arch_str(self) -> str:
        if isinstance(self.features.cells, DARTSCells):
            return self.features.cells.arch_str()
        else:
            raise TypeError(f'Cells are not DARTSCells but {type(self.features.cells)}')


class STU_NATSbench(ImageModel):
    r"""NATS-Bench proposed by Xuanyi Dong from University of Technology Sydney.

    :Available model names:

        .. code-block:: python3

            ['nats_bench']

    Note:
        There are prerequisites to use the benchmark:

        * ``pip install nats_bench``.
        * ``git clone https://github.com/D-X-Y/AutoDL-Projects.git`` and ``pip install .``
        * Extract ``NATS-tss-v1_0-3ffb9-full`` to :attr:`nats_path`.

    See Also:

        * paper: `NATS-Bench\: Benchmarking NAS Algorithms for Architecture Topology and Size`_
        * code:

          - AutoDL: https://github.com/D-X-Y/AutoDL-Projects
          - NATS-Bench: https://github.com/D-X-Y/NATS-Bench

    Args:
        model_index (int): :attr:`model_index` passed to
            ``api.get_net_config()``.
            Ranging from ``0 -- 15624``.
            Defaults to ``0``.
        model_seed (int): :attr:`model_seed` passed to
            ``api.get_net_param()``.
            Choose from ``[777, 888, 999]``.
            Defaults to ``777``.
        hp (int): Training epochs.
            :attr:`hp` passed to ``api.get_net_param()``.
            Choose from ``[12, 200]``.
            Defaults to ``200``.
        nats_path (str): NATS benchmark file path.
            It should be set as format like
            ``'**/NATS-tss-v1_0-3ffb9-full'``
        search_space (str): Search space of topology or size.
            Choose from ``['tss', 'sss']``.
        dataset_name (str): Dataset name.
            Choose from ``['cifar10', 'cifar10-valid', 'cifar100', 'imagenet16-120']``.

    .. _NATS-Bench\: Benchmarking NAS Algorithms for Architecture Topology and Size:
        https://arxiv.org/abs/2009.00437
    """
    available_models = ['stu_nats_bench']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--model_index', type=int)
        group.add_argument('--model_seed', type=int)
        group.add_argument('--hp', type=int)
        group.add_argument('--nats_path')
        group.add_argument('--search_space')
        return group

    def __init__(self, name: str = 'nats_bench', model: type[_NATSbench] = _NATSbench,
                 model_index: int = 0, model_seed: int = 777, hp: int = 200,
                 dataset: ImageSet | None = None, dataset_name: str | None = None,
                 nats_path: str | None = None,
                 search_space: str = 'tss', **kwargs):
        try:
            # pip install nats_bench
            from nats_bench import create   # type: ignore
            from xautodl.models import get_cell_based_tiny_net   # type: ignore
        except ImportError:
            raise ImportError('You need to install nats_bench and auto-dl library')

        if isinstance(dataset, ImageSet):
            kwargs['dataset'] = dataset
            if dataset_name is None:
                dataset_name = dataset.name
            if dataset_name == 'imagenet16':
                dataset_name = f'imagenet16-{dataset.num_classes:d}'
        assert dataset_name is not None
        dataset_name = dataset_name.replace('imagenet16', 'ImageNet16')
        self.dataset_name = dataset_name

        self.model_index = model_index
        self.model_seed = model_seed
        self.hp = hp
        self.search_space = search_space
        self.nats_path = nats_path

        self.api = create(nats_path, search_space, fast_mode=True, verbose=False)
        config: dict[str, Any] = self.api.get_net_config(model_index, dataset_name)
        self.get_cell_based_tiny_net: Callable[..., nn.Module] = get_cell_based_tiny_net
        network = self.get_cell_based_tiny_net(config)
        super().__init__(name=name, model=model, network=network, **kwargs)
        self.param_list['nats_bench'] = ['arch_str', 'model_index', 'model_seed', 'hp', 'search_space', 'nats_path']
        self._model: _NATSbench

    @property
    def arch_str(self) -> str:
        if isinstance(self._model.features.cells, DARTSCells):
            return self._model.arch_str()
        config = self.api.get_net_config(self.model_index, self.dataset_name)
        return config['arch_str']

    def val_loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, reduction: str = 'batchmean', **kwargs) -> torch.Tensor:

        return super().loss(_input, _label, _output, reduction, **kwargs)


    # def arch_parameters(self) -> list[torch.Tensor]:
    #     return self._model.features.arch_parameters()

    # def named_arch_parameters(self) -> list[tuple[str, torch.Tensor]]:
    #     return self._model.features.named_arch_parameters()
    
    def get_parameter_from_name(self, name: str = 'full'
                                ) -> Iterator[nn.Parameter]:
        match name:
            case 'features':
                params = self._model.features.parameters()
            case 'classifier' | 'partial':
                params = self._model.classifier.parameters()
            case 'full':
                params = itertools.chain(self._model.parameters(), self._model.arch_parameters())
            case _:
                raise NotImplementedError(f'{name=}')
        return params
    
    
    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None, _soft_label: torch.Tensor = None,
             _output: torch.Tensor = None, amp: bool = False, reduction: str = 'batchmean',**kwargs) -> torch.Tensor:
        if _output is None:
            _output = self(_input, **kwargs)
        if _soft_label is None:
            # print("validate")
            return self.val_loss(_input=_input, _label=_label, _output=_output, amp=amp)
        temp = 5.0
        criterion = nn.KLDivLoss(reduction='batchmean')
        # print("KLDivLoss")
        if amp:
            with torch.cuda.amp.autocast():
                return criterion(F.log_softmax(_output/temp,dim=1),F.softmax(_soft_label/temp,dim=1))
        return criterion(F.log_softmax(_output/temp,dim=1),F.softmax(_soft_label/temp,dim=1))

    def get_official_weights(self, model_index: int | None = None,
                             model_seed: int | None = None,
                             hp: int | None = None,
                             **kwargs) -> OrderedDict[str, torch.Tensor]:
        model_index = model_index if model_index is not None else self.model_index
        model_seed = model_seed if model_seed is not None else self.model_seed
        hp = hp if hp is not None else self.hp
        _dict: OrderedDict[str, torch.Tensor] = self.api.get_net_param(
            model_index, self.dataset_name, model_seed, hp=str(hp))
        if _dict is None:
            raise FileNotFoundError(f'Loaded weight is None. Please check {self.nats_path=}.\n'
                                    'It should be set as format like "**/NATS-tss-v1_0-3ffb9-full"``')
        new_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in _dict.items():
            if k.startswith('stem') or k.startswith('cells') or k.startswith('lastact'):
                new_dict['features.' + k] = v
            elif k.startswith('classifier'):
                new_dict['classifier.fc' + k[10:]] = v
        return new_dict

    # def genotype(self) -> Genotype:
    #     return self._model.features.genotype() if self.supernet else self._model.features.genotype

    def _distillation(self, epochs: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
               adv_train: bool = None,
               lr_warmup_epochs: int = 0,
               model_ema: ExponentialMovingAverage = None,
               model_ema_steps: int = 32,
               grad_clip: float = None, pre_conditioner: None | KFAC | EKFAC = None,
               print_prefix: str = 'Epoch', start_epoch: int = 0, resume: int = 0,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               loader_train: torch.utils.data.DataLoader = None,
               loader_valid: torch.utils.data.DataLoader = None,
               epoch_fn: Callable[..., None] = None,
               get_data_fn: Callable[...,
                                     tuple[torch.Tensor, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               after_loss_fn: Callable[..., None] = None,
               validate_fn: Callable[..., tuple[float, float]] = None,
               save_fn: Callable[..., None] = None, file_path: str = None,
               folder_path: str = None, suffix: str = None,
               writer=None, main_tag: str = 'train', tag: str = '',
               accuracy_fn: Callable[..., list[float]] = None,
               verbose: bool = True, indent: int = 0, tea_forward_fn: Callable[..., torch.Tensor] = None, **kwargs):
        get_data_fn = get_data_fn or self.get_data
        validate_fn = validate_fn or self._dis_validate


        get_data_old = get_data_fn
        validate_old = validate_fn

        def get_data(data: tuple[torch.Tensor, torch.Tensor], adv_train: bool = False,
                        mode: str = 'train_STU', **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
            # for test
            # if self.count > 0 and mode != 'valid':
            #     mode = 'train_STU'
            #     self.count -= 1
            # elif self.count == 0 and mode != 'valid':
            #     mode = 'train_GEN'
            #     self.count = 1

            if mode == 'train_STU' or mode == 'train_ADV_STU':
                _input, _label = get_data_old(data, adv_train=adv_train, **kwargs)
                _soft_label = tea_forward_fn(_input,**kwargs)
                _soft_label.detach()
                # _output = self(_input, **kwargs)
                return _input, _label, _soft_label
            elif mode =='valid':
                _input, _label = get_data_old(data, adv_train=adv_train, **kwargs)
                return _input, _label

        def fun(variable):
            num = ['0','1','2','3','4','5','6','7','8','9','']
            if (variable in num):
                return False
            else:
                return True
    
        def _validate(adv_train: bool = None,
                        loader: torch.utils.data.DataLoader = None,
                        tea_forward_fn: Callable[..., torch.Tensor] = None,
                        **kwargs) -> tuple[float, float]:
            # print(self.genotype)
            # stu_arch_list = list(filter(None, re.split('\+|\|',self.arch_str)))
            stu_arch_tensor = self._model.arch_parameters()[0]
            stu_arch_tensor = F.normalize(stu_arch_tensor, p=2, dim=1)
            return validate_old(loader=loader, adv_train=adv_train, stu_arch_tensor=stu_arch_tensor ,tea_forward_fn=tea_forward_fn,**kwargs)

        get_data_fn = get_data
        validate_fn = _validate

        return super()._distillation(epochs=epochs, optimizer=optimizer, lr_scheduler=lr_scheduler,
                            adv_train=adv_train,
                            lr_warmup_epochs=lr_warmup_epochs,
                            model_ema=model_ema, model_ema_steps=model_ema_steps,
                            grad_clip=grad_clip, pre_conditioner=pre_conditioner,
                            print_prefix=print_prefix, start_epoch=start_epoch,
                            resume=resume, validate_interval=validate_interval,
                            save=save, amp=amp,
                            loader_train=loader_train, loader_valid=loader_valid,
                            epoch_fn=epoch_fn, get_data_fn=get_data_fn,
                            loss_fn=loss_fn, after_loss_fn=after_loss_fn,
                            validate_fn=validate_fn,
                            save_fn=save_fn, file_path=file_path,
                            folder_path=folder_path, suffix=suffix,
                            writer=writer, main_tag=main_tag, tag=tag,
                            accuracy_fn=accuracy_fn,
                            verbose=verbose, indent=indent, tea_forward_fn=tea_forward_fn, **kwargs)
