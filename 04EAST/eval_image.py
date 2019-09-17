# This program evaluate single image, return bounding boxes of texts detected
import cv2
import time
import math
import os
import numpy as np
import model
import tensorflow as tf
import eval as ev
import locality_aware_nms as nms_locality
import lanms

checkpoint_path = "tmp/east_icdar2015_resnet_v1_50_rbox/"

with tf.get_default_graph().as_default():
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

def get_boxes(img):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        saver.restore(sess, model_path)

        img = cv2.imread(img)[:, :, ::-1]
        img_resized, (ratio_h, ratio_w) = ev.resize_image(img)
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [img_resized]})
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        boxes, timer = ev.detect(score_map=score, geo_map=geometry, timer=timer)

        res = []
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            for box in boxes:
                # to avoid submitting errors
                box = ev.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                res.append([box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]])
        return res

def get_all_boxes(img):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        saver.restore(sess, model_path)

        img = cv2.imread(img)[:, :, ::-1]
        img_resized, (ratio_h, ratio_w) = ev.resize_image(img)
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [img_resized]})
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        boxes, timer = ev.detect(score_map=score, geo_map=geometry, timer=timer)

        res = []
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            for box in boxes:
                # to avoid submitting errors
                box = ev.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                res.append([box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]])
        return res

def resize(folder, name, boxes):
    img = cv2.imread(folder + "/" + name)
    h, w = img.shape[:2]
    idx = 0
    for res in boxes:
        xList = [res[0], res[2], res[4], res[6]]
        yList = [res[1], res[3], res[5], res[7]]
        xmin, ymin, xmax, ymax = (min(xList), min(yList), max(xList), max(yList))
        ch = ymax - ymin
        cw = xmax - xmax
        ratio = 0.0
        xmin = int(max(0, xmin - ratio/2 * cw))
        xmax = int(min(w, xmax + ratio/2 * cw))
        ymin = int(max(0, ymin - ratio/2 * ch))
        ymax = int(min(h, ymax + ratio/2 * ch))
        crop_img = img[ymin:ymax, xmin:xmax]
        newfilename = folder + "/croped/" + name.split(".")[0] + "-crop" + str(idx) + ".jpg"
        print(newfilename)
        cv2.imwrite(newfilename, crop_img)
        idx += 1

def get_croped_images(folder, img_files):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        saver.restore(sess, model_path)
        for img_file in img_files:
            filename = folder + "/" + img_file
            print(filename)
            img = cv2.imread(filename)[:, :, ::-1]
            img_resized, (ratio_h, ratio_w) = ev.resize_image(img)
            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [img_resized]})
            timer = {'net': 0, 'restore': 0, 'nms': 0}
            boxes, timer = ev.detect(score_map=score, geo_map=geometry, timer=timer)

            res = []
            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                for box in boxes:
                    # to avoid submitting errors
                    box = ev.sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    res.append([box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]])

            if len(res) > 0:
                resize(folder, img_file, res)

directory = "tmp/result"
files = []
for file in os.listdir(directory):
    if file.endswith(('.jpeg', '.png', '.jpg')):
        files.append(file)

get_croped_images(directory, files)





