# Short description of meta files

## train_metadata_extended_noduplv1.csv

Contains original train metadata with some additional meta. Also manually observed duplicates were dropped

TODO: Drop duplicates according to [next discussion](https://www.kaggle.com/competitions/birdclef-2024/discussion/494134)

## full_noduplsV3_scored_meta_prev_comps_extended_2024SecLabels.csv

Contains info about audio files, which belong to target classes, from previous BirdCLEF competitions. Also duplicates from `train_metadata_extended_noduplv1.csv` were dropped

In order to match with wave files from this [dataset](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2024-add-data) - you have to create `id` column

```python
df["id"] = df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
```

__IMPORTANT__: Do not use original `id` from table !!!

## train_metadata_noduplV3_extended_2024SecLabels.csv

Contains info about audio files, which belong to target classes, from Xeno Canto. Also duplicates from `train_metadata_extended_noduplv1.csv` and `full_noduplsV3_scored_meta_prev_comps_extended_2024SecLabels` were dropped

In order to match with wave files from this [dataset](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2024-add-data) - you have to create `id` column

```python
df["id"] = df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
```

__IMPORTANT__: Do not use original `id` from table !!!

## train_data_google_bird_model_pseudo.csv

Contains __logit__ predictions from [bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8), obtained using [notebook](https://www.kaggle.com/code/vladimirsydor/bird-clef-2024-google-clf/notebook), for all train data. In order to get probabilities - apply sigmoid

__IMPORTANT__: Model does not predict `bkrfla1` and `indrol2` but I have added `bkrfla2`, `indrol1` and `indrol3`. __DO NOT THINK__ that these classes are equal to original, from my observations it is not true

In order to match with wave files from this train dataset - you have to create `id` column

```python
df["id"] = df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
```

__IMPORTANT__: Do not use original `id` from table !!!

## add_scored_data_google_bird_model_pseudo.csv

Contains __logit__ predictions from [bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8), obtained using [notebook](https://www.kaggle.com/code/vladimirsydor/bird-clef-2024-google-clf/notebook), for `full_noduplsV3_scored_meta_prev_comps_extended_2024SecLabels.csv` and `train_metadata_noduplV3_extended_2024SecLabels.csv`. In order to get probabilities - apply sigmoid

__IMPORTANT__: Model does not predict `bkrfla1` and `indrol2` but I have added `bkrfla2`, `indrol1` and `indrol3`. __DO NOT THINK__ that these classes are equal to original, from my observations it is not true

In order to match with wave files from this [dataset](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2024-add-data) - you have to create `id` column

```python
df["id"] = df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
```

__IMPORTANT__: Do not use original `id` from table !!!

## unlabeled_data_google_bird_model_pseudo.csv

Contains __logit__ predictions from [bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8), obtained using [notebook](https://www.kaggle.com/code/vladimirsydor/bird-clef-2024-google-clf/notebook), for unlabeled data. In order to get probabilities - apply sigmoid

__IMPORTANT__: Model does not predict `bkrfla1` and `indrol2` but I have added `bkrfla2`, `indrol1` and `indrol3`. __DO NOT THINK__ that these classes are equal to original, from my observations it is not true

All predictions are made in submission format
