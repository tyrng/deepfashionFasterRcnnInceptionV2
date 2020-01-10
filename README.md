# DeepFashion Fashion Detection using Tensorflow Object Detection API

Requires Python 3. Tested on Python 3.7.6 in macOS 10.13.6 and 10.14.5.

## Installation:
1. Clone [Tensorflow Models Repository](https://github.com/tensorflow/models) and put inside tf/ folder (tf/models/)
2. Follow the setup for [Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). Remember to specify tensorflow==1.14 or tensorflow-gpu==1.14, tensorflow 2.0 does not work for this API.
3. pip install --requirement requirements.txt (for extra dependencies used in this repo)

## Recommended Directory Structure:

```
+accuracy_confusionMatrix
  -accuracy_confusionMatrix.py
+data
  -label_map file
  -train TFRecord file
  -test TFRecord file
  -eval TFRecord file
+docs
+generate_tfrecord
  -deep_fashion_to_tfrecord.py
+inference
  -Object_detection_image.py
  -Object_detection_webcam.py
+models
  + <model name>
    -pipeline config file
    +train
    +eval
+tf
  +models (tensorflow models repository)
-requirements.txt
```

## Preparing Environment Variables
```bash
export PYTHONPATH=$PYTHONPATH:<path to project directory>/tf/models/research:<path to project directory>/tf/models/research/slim
```
Remember to run this in every terminal session prior to doing anything else. Or you can put this in your .bashrc (.bash_profile for macOS) file for persistency.

## Dataset Preparation:

```bash
# From the project root directory
DATASET_PATH={Path to DeepFashion project dataset with Anno, Eval and Img directories e.g. /home/user/deepfashion/}
OUTPUT_PATH={Path to output TFRecord e.g. data/val.record}
CATEGORIES={broad or fine, broad FOR top, bottom or full only, fine FOR categories.}
EVALUATION_STATUS={train, val or Test}
LABEL_MAP_PATH={Path to label map proto, e.g. data/deepfashion_label_map_fine.pbtxt.}

python generate_tfrecord/deep_fashion_to_tfrecord.py \
    --dataset_path ${DATASET_PATH} \
    --output_path ${OUTPUT_PATH} \
    --categories ${CATEGORIES} \
    --evaluation_status ${EVALUATION_STATUS} \
    --label_map_path ${LABEL_MAP_PATH}
```

## Training and Evaluation:

1. Training and evaluation using the sample configs:
```bash
# From the tensorflow/models/research/ directory
cd tf/models/research/

PIPELINE_CONFIG_PATH={path to pipeline config file e.g. models/<model name>/pipeline.config}
MODEL_DIR={path to model directory e.g. models/<model name>/}
NUM_TRAIN_STEPS={60000 was used FOR SsdResnet50, 3000000 was used FOR fasterRcnnInceptionV2}
SAMPLE_1_OF_N_EVAL_EXAMPLES=50

python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```
2. Running Tensorboard:
```bash
MODEL_DIR={path to model directory e.g. models/<model name>/}
tensorboard --logdir=${MODEL_DIR}
```
## Exporting a Trained Model:

``` bash
# From tensorflow/models/research/
cd tf/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH={path to pipeline config file e.g. models/<model name>/pipeline.config}
TRAINED_CKPT_PREFIX={path to model.ckpt e.g. models/<model name>/model.ckpt-<CHECKPOINT_NUMBER>}
EXPORT_DIR={path to folder that will be used For Export e.g. models/<model name>/inference_graph }

python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

## Generating Top-N Accuracy and Confusion Matrix:
```bash
# From the project root directory

CKPT_PATH={Path to frozen detection graph .pb file, Which contains the model that is used e.g. models/<model name>/inference_graph/frozen_inference_graph.pb}
LABEL_MAP_PATH={Path to label map proto, e.g. data/deepfashion_label_map_fine.pbtxt.}
NUM_CLASSESS={Number of classes the labels have. e.g. 50 FOR DeepFashion}
TFRECORD_PATH={Should use evaluation Set .e.g data/eval.record}
SKIP_EVERY={Same as sample_1_of_n_eval_examples, skip every n images during evaluation. e.g. 50 was used In the paper}
TOP={Top N accuracy needed. e.g. 1 or 3 were used In the paper}
MINIMUM_SCORE={Threshold FOR minimum score that is considered a True positive. e.g. 0.3}
OUTPUT_RESULT={Path to a text file containing all the results}
CONFUSION_TOPN={True or False FOR whether the confusion matrix should follow top N results.}

python accuracy_confusionMatrix/accuracy_confusionMatrix.py \
    --ckpt_path ${CKPT_PATH} \
    --label_map_path ${LABEL_MAP_PATH} \
    --num_classes ${NUM_CLASSESS} \
    --tfrecord_path ${TFRECORD_PATH} \
    --skip_every ${SKIP_EVERY} \
    --top ${TOP} \
    --minimumScore ${MINIMUM_SCORE} \
    --outputResult ${OUTPUT_RESULT} \
    --confusion_topn ${CONFUSION_TOPN}
```

## Inferencing on Webcam:
```bash
# From the project root directory

CKPT_PATH={Path to frozen detection graph .pb file, Which contains the model that is used e.g. models/<model name>/inference_graph/frozen_inference_graph.pb}
LABEL_MAP_PATH={Path to label map proto, e.g. /ata/deepfashion_label_map_fine.pbtxt.}
NUM_CLASSESS={Number of classes the labels have. e.g. 50 FOR DeepFashion}
MIN_SCORE={Threshold FOR minimum score that is considered a True positive. e.g. 0.3}

python inference/Object_detection_webcam.py \
    --ckpt_path ${CKPT_PATH} \
    --label_map_path ${LABEL_MAP_PATH} \
    --num_classes ${NUM_CLASSESS} \
    --min_score ${MIN_SCORE}
```

## Inferencing on an Image:
```bash
# From the project root directory

CKPT_PATH={Path to frozen detection graph .pb file, Which contains the model that is used e.g. models/<model name>/inference_graph/frozen_inference_graph.pb}
LABEL_MAP_PATH={Path to label map proto, e.g. data/deepfashion_label_map_fine.pbtxt.}
NUM_CLASSESS={Number of classes the labels have. e.g. 50 FOR DeepFashion}
MIN_SCORE={Threshold FOR minimum score that is considered a True positive. e.g. 0.3}
IMAGE_PATH={Path to an image, a png file is used In the paper}

python inference/Object_detection_image.py \
    --ckpt_path ${CKPT_PATH} \
    --label_map_path ${LABEL_MAP_PATH} \
    --num_classes ${NUM_CLASSESS} \
    --min_score ${MIN_SCORE} \
    --image_path ${IMAGE_PATH}
```
