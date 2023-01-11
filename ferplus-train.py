# fer0000000.png,"(0, 0, 48, 48)",4,0,0,1,3,2,0,0,0,0

import os
import re
import shutil


label_f = open("/data/jc/data/image/FERPP/FER2013Train/label.csv", "r")
path = "/data/jc/data/image/FERPP/FER2013Train"

line = label_f.readline()
num = 0
while line:
    line = re.split(',"|",|\n',line)
    img = line[0]
    label = re.split(',',line[2])
    if label.count('10')>0:
            true_label = label.index('10')
            newpath = os.path.join(path,str(true_label))
            shutil.move(os.path.join(path,img),newpath)
            print("mv",img)
    line = label_f.readline()
    
label_f.close()
