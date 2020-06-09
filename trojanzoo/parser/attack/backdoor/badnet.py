# -*- coding: utf-8 -*-

from ..attack import Parser_Attack


class Parser_BadNet(Parser_Attack):
    attack = 'badnet'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--target_class', dest='target_class', type=int,
                            help='target class of backdoor, defaults to config[badnet][target_class][dataset]=0')
        parser.add_argument('--percent', dest='percent', type=float,
                            help='malicious training data injection probability for each batch, defaults to config[badnet][target_class][dataset]=0.1')
