#!/bin/bash

echo "GPU $1 will be used for training"

# Download data and prepare
cd data
# 2023 year data
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part2
kaggle datasets download -d vladimirsydor/birdclef-2023-data-part3
# 2025 year data
kaggle competitions download -c birdclef-2025
kaggle datasets download vladimirsydor/bird-clef-2025-add-data
kaggle datasets download vladimirsydor/bird-clef-2025-all-pretrained-models
kaggle datasets download vladimirsydor/bird-clef-2025-pseudo

# process 2023 data
unzip birdclef-2023-data-part2.zip -d birdclef_2023_data_part2
rm birdclef-2023-data-part2.zip
unzip birdclef-2023-data-part3.zip -d birdclef_2023_data_part3
rm birdclef-2023-data-part3.zip

mv birdclef_2023_data_part2/soundscapes_nocall.zip.part-* ./
mv birdclef_2023_data_part3/soundscapes_nocall.zip.part-* ./
cat soundscapes_nocall.zip.part-* > soundscapes_nocall.zip
rm soundscapes_nocall.zip.part-*
unzip soundscapes_nocall.zip
rm soundscapes_nocall.zip

mv birdclef_2023_data_part2/esc50/esc50 esc50

rm birdclef_2023_data_part2 -r
rm birdclef_2023_data_part3 -r

# process 2025 data
unzip birdclef-2025.zip -d birdclef_2025
rm birdclef-2025.zip
unzip bird-clef-2025-add-data.zip -d birdclef_2025_add_data
rm bird-clef-2025-add-data.zip
unzip bird-clef-2025-all-pretrained-models.zip -d birdclef_2025_all_pretrained_models
rm bird-clef-2025-all-pretrained-models.zip
unzip bird-clef-2025-pseudo.zip -d birdclef_2025_pseudo
rm bird-clef-2025-pseudo.zip

mv birdclef_2025/train_audio ./
mv birdclef_2025_add_data/add_train_audio_from_prev_comps ./
mv birdclef_2025_add_data/add_train_audio_from_xeno_canto_28032025 ./

rm birdclef_2025_add_data -r

cd ../

python scripts/precompute_features.py data/train_audio data/train_features --n_cores 8
python scripts/precompute_features.py data/add_train_audio_from_prev_comps data/add_train_features_from_prev_comps --n_cores 8 --use_torchaudio
python scripts/precompute_features.py data/add_train_audio_from_xeno_canto_28032025 data/add_train_features_from_xeno_canto_28032025 --n_cores 8 --use_torchaudio
python scripts/precompute_features.py data/birdclef_2025/train_soundscapes data/train_features_soundscapes --n_cores 8



# Start training
WANDB_MODE="offline" CUDA_VISIBLE_DEVICES="$1" python scripts/main_train.py train_configs/selected_ebs.py

WANDB_MODE="offline" CUDA_VISIBLE_DEVICES="$1" python scripts/main_train.py train_configs/selected_eca.py

# Run inference and compile models
CUDA_VISIBLE_DEVICES="$1" python scripts/main_inference_and_compile.py inference_configs/selected_ebs.py

CUDA_VISIBLE_DEVICES="$1" python scripts/main_inference_and_compile.py inference_configs/selected_eca.py
