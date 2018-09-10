# -*- coding: utf-8 -*-
"""
Created on Sep 10 11:53:19 2018

Input arguments:
video_path = sys.argv[1] should be the input file path

Output parameters:
pedestrian_number

A text file of:
(1) detected bounding boxes
(2) frame IDs (for video)


@author: kuanyew
"""

import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import csv
import glob
import os
import re

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the pre-trained model expects images to have shape
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


# Function to set regions on mask areas
def get_block_regions():

    '''Initialize two block regions'''
    a = 650
    region_1 = np.array([[0, 0], [a, 0], [0, 350]], np.int32)
    region_2 = np.array([[0, 360], [660, 360], [768, 432], [0, 432]], np.int32)
    region_3 = np.array([[768, 432], [350, 160], [a, 0], [768, 0]], np.int32)

    return region_1, region_2, region_3

pat=re.compile("(\d+)\D*$")

# Function to sort the image sequences in the directory
def key_func(x):
    mat = pat.search(os.path.split(x)[-1])  # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1))


if __name__ == "__main__":
    model_path = 'tensorflow_model/mask_rcnn_inception_v2_coco_2018.pb'
    pedestrian_model = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.6

    input_path = sys.argv[1]

    region_1, region_2, region_3 = get_block_regions()

    out = cv2.VideoWriter('Pedestrian_demo_01.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (768, 432))

    with open('result.csv', 'w', newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        frame_counter = 1
        smooth_control = []

        cap = cv2.VideoCapture(input_path)
        all_images = input_path + '/*.png'

        for fn in sorted(glob.glob(all_images), key=key_func):
            img = cv2.imread(fn)
            img = cv2.resize(img, (768, 432))

            # Draw block regions
            frame_transparent = img.copy()
            cv2.fillPoly(frame_transparent, [region_1, region_2, region_3], color=(0, 255, 0), lineType=cv2.LINE_AA)
            img2 = cv2.addWeighted(img, 0.5, frame_transparent, 0.1, 0)

            # Detect all visible pedestrian
            boxes, scores, classes, num = pedestrian_model.processFrame(img)
            copy_box = []

            # Visualization of the results of a detection.
            detected = 0
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(img2, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
                    copy_box.append(box)
                    detected += 1

            # Detect pedestrian at the frontal pathway only
            cv2.fillPoly(img, [region_1, region_2, region_3], color=(0, 255, 0), lineType=cv2.LINE_AA)
            img3 = cv2.addWeighted(img, 0.5, img, 0.9, 0)
            boxes_frontal, scores_frontal, classes_frontal, num_frontal = pedestrian_model.processFrame(img3)
            detect_front = 0
            for i in range(len(boxes_frontal)):
                # Class 1 represents human
                if classes_frontal[i] == 1 and scores_frontal[i] > threshold:
                    detect_front += 1

            # Output results
            result_writer.writerow([frame_counter] + [detected])
            result_writer.writerow([copy_box])
            frame_counter += 1

            cv2.putText(img2, text='Pedestrian along the vehicle\'s frontal pathway: {}'.format(
                detect_front), org=(50, 400),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5, color=(255, 255, 255),
                        thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img2, text='Total of pedestrian detected: {}'.format(
                detected), org=(50, 420),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5, color=(255, 255, 255),
                        thickness=1, lineType=cv2.LINE_AA)

            out.write(img2)
            cv2.imshow("preview", img2)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    out.release()
    cv2.destroyAllWindows()