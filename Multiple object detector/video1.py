import tensorflow as tf
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import subprocess
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
def execute(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (result, error) = process.communicate()
    return result
cap = cv2.VideoCapture(0)
ret, image_np = cap.read()

from utils import label_map_util

from utils import visualization_utils as vis_util
timer=0
a=0
location=[0,0]
ra=0
ck=0
MODEL_NAME = '2'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('2', 'label_map.pbtxt')
lf=0
NUM_CLASSES = 2
def find_center(a):
  X,Y = ( np.average(a[:2]),np.average(a[2:]))
  return X,Y
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while ret:
      x=1
      ret, image_np = cap.read()
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=5)
      if (np.squeeze(scores)[0]>0.8):#print coordinate value of the bounding box
          print(find_center(np.squeeze(boxes)[0]))
          location[a]=find_center(np.squeeze(boxes)[0])#find the center of the object
          image_np[0,0]=[0,0,255]
          a=1
          if(location[a]!=0):
              if((location[a][0] < location[0][0]+0.01) and (location[a][0] > location[0][0]-0.01)):#define a center
                  print("center")
                  ck=0
                  timer=0
              elif(location[a][0]>location[0][0]):
                  print("left")
                  if (ra==10 and ck==0):
                      timer+=1
                      if(timer==5):#make sure is left
                          execute("left.py")
                          ra=0
                          ck=1
                          timer=0
                  if lf!=10:#wait for 10s
                      lf+=1
              elif(location[a][0]<location[0][0]):
                  print("right")
                  if ra!=10:#wait for 10s
                      ra+=1
                  if (ra==10 and ck==0):
                      timer+=1
                      if(timer==5):#make sure is left
                          execute("right.py")
                          ck=2
                          lf=0
                          timer=0
          elif (find_center(np.squeeze(boxes)[0])==0):#when object out of the range reset a new center
            location[0]=find_center(np.squeeze(boxes)[0])
      cv2.imshow('object detection', cv2.resize(image_np, (1200,800)))
      key=cv2.waitKey(25)
      if key == 27:#use ESC to quit the video
        cv2.destroyAllWindows()
        break
