#!/bin/bash

HANNA_DATA_DIR=$(realpath ../data)
HUG_MODEL_DIR=$(realpath /home/victor/work/vlnwork/cvpr/cvpr_code/R2R-EnvDrop/pretrained_hug_models)
RESULT_MODEL_DIR=$(realpath /home/victor/work/vlnwork/cvpr/cvpr_code/hanna/possible_models)

echo "Matterport3D data: $MATTERPORT_DATA_DIR"
echo "HANNA data: $HANNA_DATA_DIR"

#nvidia-docker run -it --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/hanna/code/data/v1/scans,readonly --mount type=bind,source=$HANNA_DATA_DIR,target=/root/mount/hanna/data,readonly --volume `pwd`:/root/mount/hanna/code hanna
docker run -it --gpus '"device=0"' --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/hanna/code/data/v1/scans,readonly --mount type=bind,source=$HANNA_DATA_DIR,target=/root/mount/hanna/data,readonly --mount type=bind,source=$HUG_MODEL_DIR,target=/root/mount/hanna/hug_model,readonly --mount type=bind,source=$RESULT_MODEL_DIR,target=/root/mount/hanna/result_model --volume `pwd`:/root/mount/hanna/code hanna
