#! /bin/bash
PYTHON="/home/lilium/anaconda3/envs/tensorflow/bin/python"

ENCODER="resnet50"
DECODER="attention"
DATASET="nyu"
## experimental settings
CLASSIFIER="OR"
INFERENCE="soft"
NUM_CLASSES=80

RGB_DIR="/home/lilium/myDataset/NYU_v2/"
DEP_DIR="/home/lilium/myDataset/NYU_v2/"

TEST_USE_FLIP=Flase
TEST_USE_MS=True

WORKSPACE_DIR="../workspace/"
LOG_DIR="log_${ENCODER}${DECODER}_${DATASET}_${CLASSIFIER}"
TEST_CHECKPOINT="ResNet_002.pkl"
TEST_RESTORE_FROM="${WORKSPACE_DIR}${LOG_DIR}/${TEST_CHECKPOINT}"
# test set
TEST_RGB_TXT="../datasets/nyu_path/valid_rgb.txt"
TEST_DEP_TXT="../datasets/nyu_path/valid_depth.txt"
TEST_RES_DIR="res"
MODE="test"
$PYTHON -u depthest_main.py --mode $MODE --encoder $ENCODER --decoder $DECODER --classifier $CLASSIFIER --inference $INFERENCE --classes $NUM_CLASSES \
                            --dataset $DATASET --rgb-dir $RGB_DIR --dep-dir $DEP_DIR --test-rgb $TEST_RGB_TXT --test-dep $TEST_DEP_TXT \
                            --gpu True --use-flip $TEST_USE_FLIP --use-ms $TEST_USE_MS --logdir $LOG_DIR --resdir $TEST_RES_DIR  \
                            --resume $TEST_RESTORE_FROM 