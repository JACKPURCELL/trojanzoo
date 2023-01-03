import os
import shutil
import sys
import hapi
hapi.config.data_dir = "/home/jkl6486/HAPI"
dic = str('fer/ferplus/microsoft_fer/22-05-23')
dic_split = dic.split('/')
predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])
path="/home/jkl6486/FERPlus/data/"
for i in range(len(predictions[dic])):
    
    prefixs=["FER2013Test","FER2013Valid","FER2013Train"]
    for prefix in prefixs:
        file = os.path.join(path,prefix,predictions[dic][i]['example_id']+".png")
        if os.path.exists(file):
            label = predictions[dic][i]['predicted_label']
            newpath = os.path.join(path,prefix,str(label))
            shutil.move(file,newpath)
    # predictions[dic][i]['predicted_label']
    # predictions[dic][i]['confidence']


# # prefixs=["FER2013Test","FER2013Valid","FER2013Train"]
# prefix = "FER2013Test"
# info_lb = torch.zeros(len(self.targets) + 1,dtype=torch.long)
# info_conf = torch.zeros(len(self.targets) + 1)
# os.path.exists()
# os.path.join(path,prefix)
# os.mkdir(“file”)
# shutil.move(“oldpos”,”newpos”)