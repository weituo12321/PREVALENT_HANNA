#!/bin/bash

export PT_OUTPUT_DIR="/root/mount/hanna/result_model"
#export PT_OUTPUT_DIR="checkpoints"
config_file="configs/hanna.json"

shopt -s expand_aliases

alias python="python3 -u"
