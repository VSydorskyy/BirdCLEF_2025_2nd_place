#!/bin/bash

CONFIG_DIR="train_configs/configs_queue"
SCRIPT="scripts/main_train.py"
DEVICE="1"

# Loop over all Python config files in alphabetical order
for config in $(ls "$CONFIG_DIR"/*.py | sort); do
    echo "Launching training for: $config"
    CUDA_VISIBLE_DEVICES="$DEVICE" python "$SCRIPT" "$config"
done
