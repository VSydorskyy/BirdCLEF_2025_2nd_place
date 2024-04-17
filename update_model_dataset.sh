#!/bin/bash

echo "Exp folder: $1";
echo "Checkpoint name: $2";

# Set the optional postfix parameter; default is empty if not provided
postfix="${4:-}"

# Directory paths now include the postfix
dest_dir="/home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_models/${1}${postfix}"

# Create directories
mkdir -p "${dest_dir}"
mkdir -p "${dest_dir}/checkpoints"

# Copy checkpoints
for chkp_name in $2
do
    cp "logdirs/$1/$2/"* "${dest_dir}/checkpoints/"
done

# Navigate to the dataset directory and zip the directory
cd "/home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_models"
zip -r "${1}${postfix}.zip" "${1}${postfix}"
rm -rf "${1}${postfix}"

# Check if the third parameter is 'update' and if so, update the dataset on Kaggle
if [ "$3" = "update" ]; then
    echo "Updating Model Dataset"
    kaggle datasets version -p "./" -m "$1 model"
fi