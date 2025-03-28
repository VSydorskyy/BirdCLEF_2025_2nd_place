#!/bin/bash

BASE_DIR="/gpfs/space/projects/BetterMedicine/volodymyr1/exps/bird_clef_2025/kaggle_datasets/bird_clef_2025_code"
FOLDER_NAME="main_folder"
ZIP_NAME="$FOLDER_NAME.zip"
TARGET="$BASE_DIR/$FOLDER_NAME"

# Cleanup old files
rm -rf "$BASE_DIR/$ZIP_NAME" "$TARGET"

# Create folder and copy files
mkdir -p "$TARGET"
cp -r code_base "$TARGET/"
cp pyproject.toml poetry.lock "$TARGET/"

# Zip and cleanup
cd "$BASE_DIR"
zip -r "$ZIP_NAME" "$FOLDER_NAME"
rm -rf "$FOLDER_NAME"

# Upload to Kaggle
kaggle datasets version -p ./ -m "new code version"
