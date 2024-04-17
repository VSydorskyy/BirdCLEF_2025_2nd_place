
#!/bin/bash

rm /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code/main_folder.zip -rf
rm /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code/main_folder -rf
mkdir /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code/main_folder
cp code_base /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code/main_folder/ -r
cp pyproject.toml /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code/main_folder/
cp poetry.lock /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code/main_folder/
cd /home/vova/data/exps/birdclef_2024/kaggle_datasets/bird_clef_2024_code
zip main_folder.zip main_folder -r
rm main_folder -rf
kaggle datasets version -p ./ -m "new code version"
