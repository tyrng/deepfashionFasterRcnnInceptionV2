# Import packages
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from PIL import Image
import cv2

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
    # orig_stdout = sys.stdout
    # f = open('/Users/tyrone/Desktop/1/moreEval/tensors.txt', 'w')
    # sys.stdout = f
    # for m in op:
    #     print(m.values())
    # sys.stdout = orig_stdout
    # f.close()

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


n = 0
i = 0
for image_features in parsed_image_dataset:
  if n % SKIP_EVERY == 0:
    i = i + 1
    #image_raw = image_features['image/encoded'].numpy()
    #f = open("/Users/tyrone/Desktop/1/moreEval/output/" + str(i) + '-eval.png', 'wb')
    #f.write(image_raw)
    #f.close()
    image_reader = tf.image.decode_jpeg(image_features['image/encoded'])
    float_caster = tf.cast(image_reader, tf.float32)
    image_expanded = np.expand_dims(float_caster, axis=0)
    (scores, classes) = sess.run([detection_scores, detection_classes],feed_dict={image_tensor: image_expanded})
    print(str(i) + ' --- ', end=" ")
    print(str(image_features['image/object/class/text'].values[0].numpy()).split("'")[1])
    for x in range(0,5):
        print('{}%'.format(int(100*scores[0][x])), end=" ")
        print(category_index[classes[0][x].astype(np.int32)]['name'])
    # print ('scores: ')
    # for s in scores:
    #     q = 0
    #     for ss in s:
    #         if q < 3:
    #             print(ss)
    #         q = q + 1
    # print ('classes: ')
    # for cc in classes:
    #     q = 0
    #     for ccc in cc:
    #         if q < 3:
    #             print(category_index[ccc.astype(np.int32)])
    #         q = q + 1
    # print ('num: ')
    # for n in num:
    #     print(n.astype(np.int32))
    # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     float_caster,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0.60)

    # All the results have been drawn on image. Now display the image.
    #cv2.imshow('Object detector', float_caster)

    # Press any key to close the image
    #cv2.waitKey(0)

    # Clean up
    #cv2.destroyAllWindows()

  n = n + 1