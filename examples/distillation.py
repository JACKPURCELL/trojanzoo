#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0,1,2 python examples/distillation.py --verbose 1 --dataset cifar10 --model stu_nats_bench --model_index 1001 --model_seed 777 --batch_size 512 --lr 0.025 --lr_scheduler --lr_min 1e-3 --grad_clip 5.0 --epochs 200 --download --nats_path /data/rbp5354/nats/NATS-tss-v1_0-3ffb9-full --adv_train free --validate_interval 1 --save --official --color 
"""  # noqa: E501

import re

import numpy as np
import trojanvision
import argparse
import torch
from time import time

from trojanzoo.utils.output import output_iter

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
    
    
    # model = trojanvision.models.create(dataset=dataset, **kwargs)
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
    conf_list = []
    index_list = []
    variance_list=[]
    variance_index_list=[]
    output_list = []
    output_var_list = []

    
    kwargs['model_name'] = 'nats_bench'

    with open('arch_var_list.csv','ab') as f , open('./seed_var_list.csv', 'w') as p, open('./acc.csv', 'w') as a:
        
        for alpha in np.arange(0.1, 1.1, 0.1):
            kwargs['mixup_alpha'] = alpha
            print("alpha: ", alpha)
            flag = 0
            dataset = trojanvision.datasets.create(**kwargs)
            
            for seed,i in zip([777,888],range(2)):
                kwargs['model_seed'] = seed
                for index,j in zip([3010,3030,3050,3640,3650,3690,3700,3710,3830,3900],range(10)):
                    kwargs['model_index'] = index
                    tea_model = trojanvision.models.create(dataset=dataset, **kwargs)
                    if env['verbose']:
                        trojanvision.summary(env=env, dataset=dataset, model=tea_model)
                    acc, loss, conf, variance, output_tensor = tea_model._validate()#output data_size*10
                    print("acc",acc)
                    if flag == 0:
                        abc = torch.zeros([output_tensor.shape[0],2,10,10])#dim batch,seed,arch,class
                        flag = 1
                    abc[:,i,j,:] = output_tensor
                    
            arch_var = torch.var(abc,dim=1)#dim batch,arch,class
            arch_var = torch.mean(arch_var,dim=[0,2])
            seed_var = torch.var(abc,dim=2)
            seed_var = torch.mean(seed_var,dim=[0,2])
            np.savetxt(f, arch_var.detach().numpy(), delimiter=",") 
            np.savetxt(p, seed_var.detach().numpy(), delimiter=",") 



        f.close()    
        p.close()    


    
    #     conf_list.append(conf)
    #     variance_list.append(variance)
    # output_list
            
            
    # index_list.append(conf_list)
    # variance_index_list.append(variance_list)
    # conf_list = []
    # variance_list = []
        # print(conf_list)
        # for i in range(len(conf_list)):
        #     print(conf_list[i])

    # f = open('./index_list.txt', 'w')
    # p = open('./variance_index_list.txt', 'w')

    
    # for i in range(len(index_list)):
    #     for j in range(len(index_list[i])):
    #         f.write(str(index_list[i][j]))
    #         f.write("\n")
    #         print(index_list[i][j])
    #     print("\n")
    #     f.write("\n")
    #     for k in range(len(variance_index_list[i])):
    #         p.write(str(variance_index_list[i][k]))
    #         p.write("\n")
    #         print(variance_index_list[i][k])
    #     p.write("\n")
    #     print("\n")
    # f.close()    
    # filename = "/root/work/trojanzoo/cifar10_model.pt"
    # filename = "/home/jkl6486/trojanzoo/cifar10_model.pt"
    
    # tea_model.load(filename)
    
    # print("=====================AFTER LOAD TEACHER==================")
    # print(tea_model.genotype)
    # print("=====================AFTER LOAD TEACHER==================")
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
    # def fun(variable):
    #     num = ['0','1','2','3','4','5','6','7','8','9','']
    #     if (variable in num):
    #         return False
    #     else:
    #         return True
    # tea_arch_list = list(filter(fun, re.split('\+|\||~',tea_model.arch_str)))
    # # ['nor_conv_1x1', 'none', 'nor_conv_1x1', 'skip_connect', 'skip_connect', 'nor_conv_3x3']
    # op_list =  [
    #     "none",
    #     "skip_connect",
    #     "nor_conv_1x1",
    #     "nor_conv_3x3",
    #     "avg_pool_3x3",
    # ]
    # tea_arch_tensor = torch.zeros(6, 5).cuda()
    # for i in range(0, len(tea_arch_list)):
    #     index=op_list.index(tea_arch_list[i])
    #     tea_arch_tensor[i][index] = 1
        

    
    # trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    # # filename = "./data/model/image/cifar10/stu_nats_bench_at-free.pth"

    # # model.load(filename)

    # if env['verbose']:
    #     trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
    #     trojanvision.summary(env=env, dataset=dataset, model=tea_model)
    # print("=====================TEACHER VALIDATE==================")
    # acc, loss = tea_model._validate()
    # print("===================Start training================")
    # model._distillation(tea_forward_fn=tea_model.__call__, tea_arch_tensor=tea_arch_tensor, tea_arch_list=tea_arch_list, **trainer)

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
