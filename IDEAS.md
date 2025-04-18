# All Ideas

## TODO

- Additional undersampled data - https://www.kaggle.com/competitions/birdclef-2025/discussion/570760
- INaturalist dataset - https://github.com/visipedia/inat_sounds/tree/main/2024
- Augmentations - https://www.kaggle.com/competitions/birdclef-2025/discussion/570577
  - SpecAugment++ - https://github.com/WangHelin1997/SpecAugment-plus/blob/main/augmentation.py
- Remove voice from train data - https://www.kaggle.com/competitions/birdclef-2025/discussion/568886
- More additional data - https://bioacoustic-ai.github.io/bioacoustics-datasets/
- Postprocessing
  - https://www.kaggle.com/competitions/birdclef-2025/discussion/568479#3156996
- Previous year solutions
  - Summarizing post - https://www.kaggle.com/competitions/birdclef-2025/discussion/568479
  - Tools and selected instruments - https://www.kaggle.com/competitions/birdclef-2025/discussion/567632
  - Maybe duplicate but let it be - https://www.kaggle.com/competitions/birdclef-2025/discussion/567507
  - Maybe duplicate but let it be V2 -https://www.kaggle.com/competitions/birdclef-2025/discussion/567499
- !!! Remove Additional data with `additional_birds`
- Compare 5 Folds and Diverse Ensemble
  ```
  I'd also like to know if ensembling diverse models is better than ensembling folds. If you are up for it (when you finish the current experiments), I'd suggest the following:

  1) pick one of the folds and use the two models you already have (actually, one of them you'll have tomorrow when training with the new encoder ends)

  2) train the same fold for 3 more models with the 3 encoders using: old eca_1, old eca_3, and new eca_4. eca_1=eca_nfnet_l0_Pretrainversion1, eca_3=eca_nfnet_l0_Pretrainversion3, eca_4 the one you are pretraining now.

  3) Train the same encoders with different dimensions: in my case (256,256) worked well with the existing encoders, though it added execution time; (256,128) didn't work as well but still better for ensembling; perhaps some other option such as (144,144) will do better. I round the 2D melspec values ("spec = (spec * 255).to(torch.uint8) / 255"). A model trained without rounding got similar CV and LB but with a lot more diversity (I'm not sure why). That could be another option.

  4) Submit the 10 models and compare with the ensemble of the old and new 5x folds models.

  If it performs better, it will be a sign that single fold is better (we could try next training the 10 models using the whole data ). If not, then we may want to stick to ensembling 5 fold models. In either case, I'd try again with a different set of models to make sure the result wasn't an outlier.
  ```
- Check out Dieter background dataset - https://www.kaggle.com/datasets/christofhenkel/birdclef2021-background-noise
- Check Best MelSpec parameters - https://www.kaggle.com/competitions/birdclef-2025/discussion/573066
- Do not use Spec Standart Scaling
- Additional data from New Zeland - https://www.kaggle.com/competitions/birdclef-2025/discussion/573677

### High Priority

- INaturalist Python API - https://github.com/pyinat/pyinaturalist

### From BirdCLEF 2024

- 1st Place https://www.kaggle.com/competitions/birdclef-2024/discussion/512197
  - Remove noisy and too loud audios:
  ```
  So for ensembles, instead of folds0-4, we use fold0 and 0.8 quantile of statistics T = std + var + rms + pwr, and this worked well. Seemingly, noisy and too loud audio harms the models.
  ```
  - Denoise samples with  [Google-bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier)
    - Remove chunks which do not match with predicted label
    - If it matches with secondary label - replace primary with secondary
    - If secondary labels exist - use 0.5 for primary and evenly distribute 0.5 for all secondary labels
    - Pseudo labels from Google Bird Classifier with 0.05 coefficient
    - Tran on 10 secs and average labels across boundary chunks
  -  Spec params
     - n_fft = 1024
     - hop_length = 500
     - n_mels = 128
     - fmin = 40
     - fmax = 15000
     - power = 2
  - Backbones
    - efficientnet_b0
    - regnety_008
  - CrossEntropyLoss as training loss + Sigmoid on inference
    - Postprocessing with several chunks
- 2nd Place


## In Progress
