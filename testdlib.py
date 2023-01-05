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
face_file_path = '/home/jkl6486/trojanzoo/afraid_African_214.jpg'# 要使用的图片，图片放在当前文件夹中
# amazed_actor_579.jpg 0 16 161 354 209 38.5907 6
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

bgr_img = cv2.resize(bgr_img, (0, 0), fx=1.1, fy=1.1, interpolation=cv2.INTER_NEAREST)
# cv2.save('resize0.jpg', img_test2)



if bgr_img is None:
    print("Sorry, we could not load '{}' as an image".format(face_file_path))
    exit()

# opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
# 检测图片中的人脸,改成box
dets = detector(rgb_img, 1)
x1 = 33
y1 = 275
x2 = 100
y2 = 342
# x1 = 57
# y1 = 180
# x2 = 242
# y2 = 366
# 创建rectangle对象,四个参数分别为左上角点位置，与右下角点的位置
rectangle = dlib.rectangle(y1, x1, y2, x2)
face_img = bgr_img[ x1:x2,y1:y2, :]
cv2.imwrite('aaaab.jpg', face_img)
# 检测到的人脸数量
num_faces = len(dets)
# if num_faces == 0:
#     print("Sorry, there were no faces found in '{}'".format(face_file_path))
#     exit()

# 识别人脸特征点，并保存下来
faces = dlib.full_object_detections()
# for det in dets:
faces.append(sp(rgb_img, rectangle))

# 人脸对齐
images = dlib.get_face_chips(rgb_img, faces, size=320)
# 显示计数，按照这个计数创建窗口
image_cnt = 0
# 显示对齐结果
for image in images:
    image_cnt += 1
    cv_rgb_image = np.array(image).astype(np.uint8)# 先转换为numpy数组
    cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)# opencv下颜色空间为bgr，所以从rgb转换为bgr
    # cv2.imshow('%s'%(image_cnt), cv_bgr_image)
    cv2.imwrite('2.jpg', cv_bgr_image)
    

cv2.waitKey(0)
cv2.destroyAllWindows()
