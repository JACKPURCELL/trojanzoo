# coding: utf-8
import cv2
import dlib
import sys
import numpy as np
import os

# 获取当前路径
current_path = os.getcwd()
# 指定你存放的模型的路径，我使用的是检测68个特征点的那个模型，
# predicter_path = current_path + '/model/shape_predictor_5_face_landmarks.dat'# 检测人脸特征点的模型放在当前文件夹中
predicter_path = '/home/jkl6486/trojanzoo/shape_predictor_68_face_landmarks.dat'
face_file_path = '/home/jkl6486/trojanzoo/excited_girl_40.jpg'# 要使用的图片，图片放在当前文件夹中
# amazed_actor_579.jpg 0 16 161 354 209 38.5907 6
# afraid_African_214.jpg 0 33 275 342 100 68.6817 2
# excited_girl_40.jpg 0 160 352 544 352 68.0181 5


# 导入人脸检测模型
detector = dlib.get_frontal_face_detector()
# 导入检测人脸特征点的模型
sp = dlib.shape_predictor(predicter_path)

# 读入图片
bgr_img = cv2.imread(face_file_path)

x, y = bgr_img.shape[0:2]
 
# 显示原图
# cv2.imshow('OriginalPicture', img)
 
# 缩放到原来的二分之一，输出尺寸格式为（宽，高）

bgr_img = cv2.resize(bgr_img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
# cv2.save('resize0.jpg', img_test2)



if bgr_img is None:
    print("Sorry, we could not load '{}' as an image".format(face_file_path))
    exit()

# opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
# 检测图片中的人脸,改成box
dets = detector(rgb_img, 1)
# x1 = 201
# y1 = 428
# x2 = 387
# y2 = 613

x1 = 160
y1 = 352
x2 = 352
y2 = 544
# 创建rectangle对象,四个参数分别为左上角点位置，与右下角点的位置
x_ = int((x2-x1)*0.05)
y_ = int((y2-y1)*0.05)
x1= x1-x_
y1= y1-y_
x2 = x2+x_
y2= y2+y_
face_img = bgr_img[ x1:x2,y1:y2, :]
rectangle = dlib.rectangle(y1, x1, y2, x2)

cv2.imwrite('aaaq.jpg', face_img)
# 检测到的人脸数量
num_faces = len(dets)
# 识别人脸特征点，并保存下来
face=sp(rgb_img, rectangle)
# 人脸对齐
image_112 = dlib.get_face_chip(rgb_img, face, size=112)
image_224 = dlib.get_face_chip(rgb_img, face, size=224)


cv_rgb_image = np.array(image_112).astype(np.uint8)# 先转换为numpy数组
cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)# opencv下颜色空间为bgr，所以从rgb转换为bgr
path_112 = 
cv2.imwrite('4.jpg', cv_bgr_image)


cv2.waitKey(0)
cv2.destroyAllWindows()