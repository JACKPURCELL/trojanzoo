import hapi
import numpy as np
import re
import cv2
import dlib
import sys
import numpy as np
import os
hapi.config.data_dir = "/home/jkl6486/HAPI"
# expw = hapi.get_labels(task="fer", dataset="expw")
# img_face = image[face_box_top:face_box_bottom, face_box_left:face_box_right, :]


hapif= open('/home/jkl6486/HAPI/data/dir/tasks/fer/expw/google_fer/test.json','r+')
hapif_f = hapif.readline()
image_path = "/data/jc/data/image/EXPW/image"

def align(face):
    images = dlib.get_face_chips(rgb_img, faces, size=224)

    faces = dlib.full_object_detections()
def crop(line):
    image_name = line[0]
    face_id_in_image = line[1]
    face_box_top = line[2]
    face_box_left = line[3]
    face_box_right = line[4]
    face_box_bottom = line[5]
    face_box_cofidence = line[6]
    expression_label = line[7]
    
    example_id = image_name.split('.')[0] + '_' + face_id_in_image
    image = cv2.imread(image_path + '/' + image_name)
    face = image[face_box_top:face_box_bottom, face_box_left:face_box_right, :]
    return face, example_id, expression_label, face_box_cofidence


if __name__ == '__main__':
    f = open("/data/jc/data/image/EXPW/label/label.lst", "r")
    line = f.readline()
    num = 0
    while line:
        line = re.split(' |\n',line)
        face, example_id, expression_label, face_box_cofidence = crop(line) 
        newpath = "/data/jc/data/image/EXPW/image_class/" + expression_label + '/' + example_id + ".jpg"
        cv2.imwrite(newpath, face)
        # {"confidence":0.99991,"predicted_label":3,"example_id":"amazed_teacher_289_0"}

        num += 1
        line = f.readline()
    f.close()
    hapif.writelines(a)


    # "example_id":"awe_boss_75_0"},

    # expw_labels = 
    #sample  angry_actor_104.jpg 0 28 113 226 141 22.9362 0
    # label.lst： each line indicates an image as follows:
    # image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_cofidence expression_label

    # for expression label：
    # "0" "angry"
    # "1" "disgust"
    # "2" "fear"
    # "3" "happy"
    # "4" "sad"
    # "5" "surprise"
    # "6" "neutral"

    print("a")