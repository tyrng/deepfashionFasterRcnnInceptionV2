# Import packages
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import tkinter
import matplotlib
import matplotlib.pyplot as plt

from _io import BytesIO

# PYTHONPATH
sys.path.append("tf/models/research")
sys.path.append("tf/models/research/slim")
sys.path.append("tf/models/research/object_detection")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str,
                    help="Path to frozen detection graph .pb file, which contains the model that is used")
parser.add_argument("--label_map_path", type=str,
                    help="Path to label map proto")
parser.add_argument("--num_classes", type=int,
                    help="Number of classes the object detector can identify")
parser.add_argument("--tfrecord_path", type=str,
                    help="Path to tfrecord")
parser.add_argument("--skip_every", type=int,
                    help="Path to tfrecord")
parser.add_argument("--top", type=int,
                    help="Top n accuracy")
parser.add_argument("--minimumScore", type=float,
                    help="(float) for minimum score to be considered as valid detection")
parser.add_argument("--outputResult", type=str,
                    help="Required text file for output.")
parser.add_argument("--confusion_topn", type=bool,
                    help="Whether confusion matrix uses Top-N results, or just Top-1 results.")

args = parser.parse_args()

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = args.ckpt_path

# Path to label map file
PATH_TO_LABELS = args.label_map_path

# Path to image
PATH_TO_TFRECORD = args.tfrecord_path

# Number of classes the object detector can identify
NUM_CLASSES = args.num_classes

SKIP_EVERY = args.skip_every

OUTPUT = args.outputResult

MIN_SCORE = args.minimumScore

TOP = args.top

CONFUSION_TOPN = args.confusion_topn

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.jet):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=90)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

raw_image_dataset = tf.data.TFRecordDataset(PATH_TO_TFRECORD)

# Create a dictionary describing the features.
image_feature_description = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64)
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    # op = sess.graph.get_operations()
    # for m in op:
    #     print(m.values())

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
#detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
#num_detections = detection_graph.get_tensor_by_name('num_detections:0')


print('Please wait...')

orig_stdout = sys.stdout

y_actu = pd.Series([], name='Actual')
# y_actu.set_value(0,'None')
y_pred = pd.Series([], name='Predicted')
# y_pred.set_value(0,'None')

n = 0
i = 0
j = 0
correct = []
for g in range(0,TOP):
  correct.append(0)
for image_features in parsed_image_dataset:
  if n % SKIP_EVERY == 0:
    trigger = []
    for g in range(0,TOP):
      trigger.append(False)
    f = open(OUTPUT, 'a')
    sys.stdout = f
    image_reader = tf.image.decode_jpeg(image_features['image/encoded'])
    float_caster = tf.cast(image_reader, tf.float32)
    image_expanded = np.expand_dims(float_caster, axis=0)
    (scores, classes) = sess.run([detection_scores, detection_classes],feed_dict={image_tensor: image_expanded})
    print(i+1, 'at', n+1, end=" --- ")
    if CONFUSION_TOPN == False:
      if scores[0][0] >= MIN_SCORE:
        # y_actu.set_value(n,int(image_features['image/object/class/label'].values[0].numpy()))
        y_actu.set_value(i,str(image_features['image/object/class/text'].values[0].numpy()).split("'")[1])
        # y_pred.set_value(n,int(category_index[classes[0][x].astype(np.int32)]['id']))
        y_pred.set_value(i,str(category_index[classes[0][0].astype(np.int32)]['name']))
      else:
        y_pred.set_value(i,'None')
    print(str(image_features['image/object/class/text'].values[0].numpy()).split("'")[1])
    for x in range(0,TOP):
      print('{0:.2f}%'.format(round(100*scores[0][x],2)), end=" ")
      if scores[0][x] > 0.0:
        print(category_index[classes[0][x].astype(np.int32)]['name'])
      else:
        print('None')

      if CONFUSION_TOPN == True:
        if scores[0][x] >= MIN_SCORE:
          # y_actu.set_value(n,int(image_features['image/object/class/label'].values[0].numpy()))
          y_actu.set_value(j,str(image_features['image/object/class/text'].values[0].numpy()).split("'")[1])
          # y_pred.set_value(n,int(category_index[classes[0][x].astype(np.int32)]['id']))
          y_pred.set_value(j,str(category_index[classes[0][x].astype(np.int32)]['name']))
        # else:
        #   # y_pred.set_value(j+1,'None')

      pred = int(category_index[classes[0][x].astype(np.int32)]['id'])
      act = int(image_features['image/object/class/label'].values[0].numpy())
      if x > 0 and trigger[x-1] == True:
        trigger[x] = True
      if pred == act and scores[0][x] >= MIN_SCORE:
        trigger[x] = True
      if trigger[x] == True:
        correct[x] = correct[x] + 1
      # print(trigger)
      j = j + 1
    # print(correct)

    sys.stdout = orig_stdout
    f.close()
    i = i + 1
  n = n + 1


f = open(OUTPUT, 'a')
sys.stdout = f
accuracy = []
for x in range(0,TOP):
  accuracy.append(float(correct[x]) / i)
  print('TOP', x+1, 'ACCURACY =', '{0:.2f}%'.format(round(100*accuracy[x],2)))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000000)

df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

print(df_conf_norm)

sys.stdout = orig_stdout
f.close()


matplotlib.use('TkAgg')
matplotlib.pyplot.show(plot_confusion_matrix(df_conf_norm))