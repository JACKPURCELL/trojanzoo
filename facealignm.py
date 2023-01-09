# coding: utf-8
import re
import cv2
import dlib
import sys
import numpy as np
import os


hapi_path = '/home/jkl6486/HAPI/data_test/dir/tasks/'
dics = ['fer/expw/microsoft_fer/22-05-23',
       'fer/expw/microsoft_fer/21-02-16',
       'fer/expw/microsoft_fer/20-03-05',
       'fer/expw/google_fer/22-05-23',
       'fer/expw/google_fer/21-02-16',
       'fer/expw/google_fer/20-03-05',
       'fer/expw/facepp_fer/22-05-23',
       'fer/expw/facepp_fer/21-02-16',
       'fer/expw/facepp_fer/20-03-05',
       'fer/expw/vgg19_fer/20-03-05',
       'fer/expw/labels']

image_path = "/data/jc/data/image/EXPW/image"
predicter_path = '/home/jkl6486/trojanzoo/shape_predictor_68_face_landmarks.dat'

count = 0
fs_line=[]
for dic in dics:
    f=open(hapi_path+dic+'.json','r')
    fs_line.append(f.readline())
    f.close()

sp = dlib.shape_predictor(predicter_path)
    


def replace(example_id):
    for i in range(len(fs_line)):
        fs_line[i]=re.sub(example_id,str(count)+'.jpg',fs_line[i])

    
def align(bgr_img,rectangle,expression_label):

    # 导入检测人脸特征点的模型
    x, y = bgr_img.shape[0:2]

    # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # 识别人脸特征点，并保存下来

    face=sp(rgb_img, rectangle)
    # 人脸对齐
    image_112 = dlib.get_face_chip(rgb_img, face, size=112)
    image_224 = dlib.get_face_chip(rgb_img, face, size=224)
    path_112 = '/data/jc/data/image/EXPW/image_class_112/'+str(expression_label)+'/'+str(count)+'.jpg'
    path_224 = '/data/jc/data/image/EXPW/image_class_224/'+str(expression_label)+'/'+str(count)+'.jpg'
    
    cv_rgb_image = np.array(image_112).astype(np.uint8)# 先转换为numpy数组
    cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)# opencv下颜色空间为bgr，所以从rgb转换为bgr
    cv2.imwrite(path_112, cv_bgr_image)
    cv_rgb_image = np.array(image_224).astype(np.uint8)# 先转换为numpy数组
    cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)# opencv下颜色空间为bgr，所以从rgb转换为bgr
    cv2.imwrite(path_224, cv_bgr_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop(line):
    image_name = line[0]
    face_id_in_image = line[1]
    x1 = int(line[2])
    y1 = int(line[3])
    y2 = int(line[4])
    x2 = int(line[5])
    face_box_cofidence = float(line[6])
    expression_label = int(line[7])
    
    x_ = int((x2-x1)*0.05)
    y_ = int((y2-y1)*0.05)
    x1= x1-x_
    y1= y1-y_
    x2 = x2+x_
    y2= y2+y_
    rectangle = dlib.rectangle(y1, x1, y2, x2)
    
    example_id = image_name.split('.')[0] + '_' + face_id_in_image
    bgr_img = cv2.imread(image_path + '/' + image_name)

    return bgr_img, rectangle, example_id, expression_label, face_box_cofidence


if __name__ == '__main__':
    label_f = open("/data/jc/data/image/EXPW/label/label.lst", "r")
    line = label_f.readline()
    num = 0
    while line:
        line = re.split(' |\n',line)
        bgr_img, rectangle, example_id, expression_label, face_box_cofidence = crop(line)
        line = label_f.readline()
        if face_box_cofidence<60:
            continue
        # align(bgr_img,rectangle,expression_label)
        replace(example_id)
        count += 1
        if count%100==0:
            print(count)
        # {"confidence":0.99991,"predicted_label":3,"example_id":"amazed_teacher_289_0"}

    label_f.close()

    for i in range(len(dics)):
        f=open(hapi_path+dics[i]+'.json','w+')
        f.writelines(fs_line[i])
        f.close()
        
