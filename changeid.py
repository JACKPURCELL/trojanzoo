import hapi

image_path = "/data/jc/data/image/EXPW/image"
hapi.config.data_dir = "/home/jkl6486/HAPI" 
# dic = ['fer/rafdb/microsoft_fer/22-05-23',
#        'fer/rafdb/microsoft_fer/21-02-16',
#        'fer/rafdb/microsoft_fer/20-03-05',
#        'fer/rafdb/google_fer/22-05-23',
#        'fer/rafdb/google_fer/21-02-16',
#        'fer/rafdb/google_fer/20-03-05',
#        'fer/rafdb/facepp_fer/22-05-23',
#        'fer/rafdb/facepp_fer/21-02-16',
#        'fer/rafdb/facepp_fer/20-03-05',
#        'fer/rafdb/vgg19_fer/20-03-05',]

def replace(example_id):

    f=open('/home/wuzz/11.txt','r')
    alllines=f.readlines()
    f.close()
    f=open('/home/wuzz/11.txt','w+')
    for eachline in alllines:
        a=re.sub('hello','hi',eachline)
        f.writelines(a)
    f.close()


dic='fer/rafdb/microsoft_fer/22-05-23'
dic_split = dic.split('/')

predictions = hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])
predictions[dic][i]['example_id']amazed_teacher_289_0
# dumps 将数据转换成字符串
info_json = json.dumps(info_dict,sort_keys=False, indent=4, separators=(',', ': '))
# 显示数据类型
print(type(info_json))
f = open('info.json', 'w')
f.write(info_json)
