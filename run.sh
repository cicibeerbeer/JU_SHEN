#!/bin/bash

# Activate your conda environment (optional)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate py39_env

# Path to model and data
MODEL_PATH="runs/detect/train10/weights/best.pt"
DATA_CFG="yolobvn.yaml"

echo "Starting YOLOv8 evaluation on TRAIN set..."

# Run evaluation on training set
yolo val model=$MODEL_PATH data=$DATA_CFG split=train save=True

echo "Evaluation completed. Results saved in runs/detect/val*/"
