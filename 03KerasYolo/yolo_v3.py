# Use yolo v3 for room number detection
# Author: ZhaonanLi, zli@brandeis.edu
# Created at Feb 2nd, 2019

# I used an online labeling tool for annotating
# Reference: https://dataturks.com/

# I modified some of the code I found from Github for training and displaying our results
# Reference: https://github.com/qqwweee/keras-yolo3

# model was trained by using annotation made by our team members:
#   ZhaonanLi, Huiyan Zhang, Yixiao He

# The model was trained on my google cloud VM for 50 + 10 epochs

import os
import cv2
from kerasyolo import YOLO
from argparse import Namespace
from PIL import Image

# IMPORTANT: Set path to the project here:
path_to_project = "/Users/nicolezhang/Desktop/149-Project"

def init():
    # clean up notes
    f = open(path_to_project + "/" + "temp/doornumber/doornumber_notes.txt", "w")
    f.write("")
    f = open(path_to_project + "/" + "temp/texts/texts_notes.txt", "w")
    f.write("")

def detect_img(yolo, path_to_images=path_to_project + "/temp/frames"):
    my_images = []
    # add all images to the list
    for file in os.listdir(path_to_images):
        if file.endswith((".jpeg", ".jpg", ".png" )):
            my_images.append(file)

    for image_file in my_images:
        path = path_to_images+ "/" + image_file
        # image is frame, convert to cv2 format
        image = Image.open(path)
        boxes = yolo.get_bounding_boxes(image)
        if boxes != None and len(boxes) > 0:
            save_door_number_img(image_file, boxes)

    yolo.close_session()

def frameCapture(file):
    # Path to video file
    vidObj = cv2.VideoCapture("01_video/" + file)
    success,image = vidObj.read()
    count = 0
    while success:
        filename = "%s%s_%d.jpg" % ("temp/frames/",file.split(".")[0],count)
        print(file)
        cv2.imwrite(filename, image)
        success, image = vidObj.read()
        count += 1
    vidObj.release()

def save_door_number_img(file, boxes, path_to_images=path_to_project + "/temp/frames"):
    f = open(path_to_project + "/temp/doornumber/doornumber_notes.txt", "a")
    path_to_image = path_to_images + "/" + file
    img = cv2.imread(path_to_image)
    h, w = img.shape[:2]
    idx = 0
    for box in boxes:
        door_number_img = img[box[1]: box[3], box[0]:box[2]]
        new_file_name = path_to_project + "temp/doornumber/" + file.split(".")[0] + "_" + str(idx) + ".jpg"
        cv2.imwrite(new_file_name, door_number_img)
        f.write("%s w:%d h:%d idx:%d xmin:%d xmax:%d ymin:%d ymax:%d \n" % (file, w, h, idx, box[0], box[2], box[1], box[3]))
        idx += 1

# init()
# frameCapture("IMG_0994.MOV")
# detect_img(YOLO())

