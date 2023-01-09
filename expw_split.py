import os
from random import sample
import random
import shutil
image_path = "/data/jc/data/image/EXPW/image"
path_112 = '/data/jc/data/image/EXPW/image_class_112/'
path_224 = '/data/jc/data/image/EXPW/image_class_224/'
 
random.seed(666)
count=0
for label in range(7):
    jpg = os.listdir(path_112+str(label))
    sample_jpgs = sample(jpg, int(len(jpg)*0.2))
    print(len(sample_jpgs),len(jpg),str(label))
    for sample_jpg in sample_jpgs:
        shutil.move(path_112+str(label)+'/'+sample_jpg, path_112+'valid/'+str(label)+'/'+ sample_jpg)
        shutil.move(path_224+str(label)+'/'+sample_jpg, path_224+'valid/'+str(label)+'/'+ sample_jpg)
        count += 1
        if count%100==0:
            print(count)


