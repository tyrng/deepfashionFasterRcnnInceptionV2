# deepfashionFasterRcnnInceptionV2

Requires Python 3. Tested on Python 3.7.6.

Installation:
1. pip install --requirement requirements.txt
2. Clone [Tensorflow Models Repository](https://github.com/tensorflow/models) and put inside tf/ folder (tf/models/)
3. Follow the setup for [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

Dataset Preparation:
1. python deepfashion_tfrecord/deep_fashion_to_tfrecord2.py \
--dataset_path '<'Path to DeepFashion project dataset with Anno, Eval and Img directories e.g. /home/user/deepfashion/'>' \
--output_path '<'Path to output TFRecord e.g. /home/user/val.record'>' \
--categories '<'broad or fine, broad for top, bottom or full only, fine for categories.'>' \
--evaluation_status '<'train, val or test'>' \
--label_map_path '<'Path to label map proto, given as deepfashion_label_map_broad.pbtxt or deepfashion_label_map_broad.pbtxt in the same folder.'>'

Training and Evaluation:
1. Refer [Tensorflow Object Detection API, Running Locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md#running-the-training-job).
2. (Optional) [To use Tensorboard](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md#running-tensorboard).

Generating Confusion Matrix and Top-N Accuracy:
1. work in progress
