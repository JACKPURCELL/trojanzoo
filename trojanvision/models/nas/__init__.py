#!/usr/bin/env python3

from trojanvision.models.imagemodel import ImageModel

from .natsbench import NATSbench

from .darts import DARTS
from .tea_darts import TEA_DARTS
from .enas import ENAS
from .lanet import LaNet
from .pnasnet import PNASNet
from .proxylessnas import ProxylessNAS

__all__ = ['NATSbench', 'DARTS', "TEA_DARTS", 'ENAS', 'LaNet', 'PNASNet', 'ProxylessNAS']
# __all__ = ['NATSbench', 'DARTS', 'ENAS', 'LaNet', 'PNASNet', 'ProxylessNAS']

class_dict: dict[str, type[ImageModel]] = {
    'nats_bench': NATSbench,
    'darts': DARTS,
    'tea_darts': TEA_DARTS,
    'enas': ENAS,
    'lanet': LaNet,
    'pnasnet': PNASNet,
    'proxylessnas': ProxylessNAS,
}
