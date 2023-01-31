import torch
import hapi        
hapi_data_dir = "/home/jkl6486/HAPI"  
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Sankey
from pyecharts import options as opts        
hapi.config.data_dir = hapi_data_dir
label_num = 7
data_num=0
def gettensor(dic):
    dic_split = dic.split('/')
    predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])
    info_lb = torch.ones(40000,dtype=torch.long)*(-1)
    
    for i in range(len(predictions[dic])):
        hapi_id = int(predictions[dic][i]['example_id'].split('.')[0])
        info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
    return info_lb
               
def plot(record):
    nodes = []
    links=[]
    for i in range(label_num):
        nodes.append({'name': "A-"+str(i)})
        nodes.append({'name': "B-"+str(i)})
        
    for i in range(label_num):
            for j in range(label_num):
                links.append({'source': "A-"+str(i), 'target': "B-"+str(j), 'value': int(record[i][j])})


    sankey = Sankey()
    sankey.add(
        '',
        nodes,
        links,
        linestyle_opt=opts.LineStyleOpts(opacity = 0.5, curve = 0.5, color = "source"),
        label_opts=opts.LabelOpts(position="top"),
        node_gap = 30,
        orient="vertical",  
        
    )
    sankey.render()
    
if __name__ == '__main__':
        dic1 = gettensor('fer/expw/microsoft_fer/22-05-23')
        dic2 = gettensor('fer/expw/google_fer/22-05-23')
        record = torch.zeros((label_num,label_num),dtype=torch.long)
        
        for label_1,label_2 in zip(dic1,dic2):
            if label_1 != -1:
                record[label_1][label_2] += 1
        plot(record)