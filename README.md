# BirdCLEF+ 2025 2nd Place Solution

## Prerequisites

- Ubuntu 22.04.3 LTS x86_64
- CUDA Version: 12.4
- CUDA Driver Version: 550.144.03
- Poetry version 2.1.3
- NVIDIA GeForce RTX 4090 (or any other GPU with VRAM ≥ 10GB)
- Hard Disk: 600GB (You may need more than 1TB if you want to pre-train)
- RAM: 126GB

## Environment

Set up Kaggle credentials to download data:

```bash
export KAGGLE_USERNAME={KAGGLE_USERNAME}
export KAGGLE_KEY={KAGGLE_KEY}
```

### Setup Poetry

1. [Install Poetry](https://python-poetry.org/docs/#installation)
   - The easiest way is to use the [Official Installer guide](https://python-poetry.org/docs/#installing-with-the-official-installer).
   - Pay attention to `poetry --version`. To install the correct version, run: `curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3`. If you have already installed another version, simply change it with: `poetry self update 2.1.3`
2. Configure Poetry to create the environment in the local folder: `poetry config virtualenvs.in-project true`
3. Activate the environment: `eval $(poetry env activate)`
4. Install dependencies: `poetry install`
5. To deactivate the environment: `exit`
6. To remove the environment: `rm -rf .venv`

## Execute the Entire Pipeline

We have gathered the main steps to reproduce our best selected solution (Public: 0.925; Private: 0.928). It will:
1. Download all required additional data from Kaggle, together with pre-trained models.
2. Perform preprocessing: Convert audio files into `.hdf5` files.
3. Train 2 models of the final ensemble (1 more will be provided as a separate Kaggle notebook).
4. Run inference to compute validation metrics and convert to fp16 OpenVINO model.

As a result, you will have 2 subfolders in the `logdir` folder:
- `eca_nfnet_l0_Exp_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromXCV2Best_PseudoF2PT05MT01P04I3_MinorOverSampleV1`
- `tf_efficientnetv2_s_in21k_Exp_noamp_64bs_5sec_BasicAug_EqualBalancing_AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromPrebs1_PseudoF2PT05MT01P04I2_AddRareBirdsNoLeak`

In each folder, you can find checkpoints and logs for each fold, and also:
- `onnx_ensem_5first_folds` - model converted into ONNX
- `onnx_ensem_5first_folds_openvino_fp16` - model compressed into fp16 and converted into OpenVINO

You can find the final OpenVINO converted models in [bird-clef-2025-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-models)

```bash
bash rock_that_bird.sh "{GPU_TO_USE}" # By default: bash rock_that_bird.sh "0"
```

## Kaggle Notebooks

## Fernando Training Notebook

- ebs_426 training notebook : TODO

### Inference

- [Kaggle Best Ensemble Inference](https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-ensemble-v2-final-final?scriptVersionId=244942051)
- Kaggle Solo Model Inference

## Solution Description

- [Kaggle Discussion](https://www.kaggle.com/competitions/birdclef-2025/discussion/583699)
- Paper: TODO

## Detailed (Advanced) Code Description

### Data

- [bird-clef-2025-add-data](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-add-data) - Additional data, with 2025 competition species
- [bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models) - Pretrained backbones for fine-tuning
- [bird-clef-2025-pseudo](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pseudo) - Pseudo labels for soundscapes
- [bird-clef-2025-pretrained-metadata](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pretrained-metadata) - Metadata for pre-trained models
- [bird-clef-2025-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-models) - Trained, converted to OpenVINO, and ready for inference models
- [bird-clef-2025-code-final](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-code-final) - Codebase on Kaggle platform needed for inference
- [bird-clef-2025-addones](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-addones) - Additional packages needed for inference

When data is loaded, you can use the `precompute_features.py` script to convert audio files into `.hdf5` format.

__IMPORTANT__: If you have `.mp3` compression, use `--use_torchaudio`

```bash
python scripts/precompute_features.py {path/to/folder/with/audio/files} {path/to/save/hdf5/files} --n_cores 8 --use_torchaudio
```

### Training

To train, use the `main_train.py` script:

```bash
WANDB_MODE="offline" CUDA_VISIBLE_DEVICES="0" python scripts/main_train.py train_configs/{your_favourite_config}.py
```

You can enable [W&B](https://wandb.ai/site/) logging if you want.

Available configs:
- `selected_ebs.py` - `tf_efficientnetv2_s_in21k` for best selected ensemble
- `selected_eca.py` - `eca_nfnet_l0` for best selected ensemble
- `best_solo_ebs.py` - `tf_efficientnetv2_s_in21k` for best solo model
- `best_ensem_ebs1.py` - `tf_efficientnetv2_s_in21k` for best Private NOT selected ensemble
- `best_ensem_ebs2.py` - `tf_efficientnetv2_s_in21k` for best Private NOT selected ensemble

### Pretraining

We have not added all audio files used for pre-training, because it is an enormous amount of data and the overall pre-trained checkpoints are quite universal, so you can reuse them for different species subsets.

Still, if you want to train pre-trained models, we have [Kaggle Dataset with metadata](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pretrained-metadata):
- `train_and_prev_comps_extendedv1_pruneSL_XCallyearstaxonomy_snipet03042025_hdf5_nosmall10sp_no2025.csv` - smaller pre-train dataset without CSA and some other datasets
- `cv_split_20_folds_train_and_prev_comps_extendedv1_pruneSL_XCallyearstaxonomy_snipet11052025_csa_newzealand_XCshiro_nosmall10sp.npy` - bigger pre-train dataset with CSA and other datasets

To obtain audio files, you can use Xeno-Canto (`scripts/download_all_xeno_canto.py`), INaturalist (`scripts/download_inaturalist.py`), and [CSA](https://colecciones.humboldt.org.co/sonidos/visor-csa/) APIs for downloading them.

After that, you can pre-train using the same script as in [Training](#training) but with the following configs:
- `pretrain_ebs_2025.py` - `tf_efficientnetv2_s_in21k` pre-trained on the smaller pre-train dataset
- `pretrain_eca_2025.py` - `eca_nfnet_l0` pre-trained on the bigger pre-train dataset
- `pretrain_ebs_2025_extended.py` - `tf_efficientnetv2_s_in21k` pre-trained on the bigger pre-train dataset
- `pretrain_eca_2025_extended.py` - `eca_nfnet_l0` pre-trained on the bigger pre-train dataset

Finally, use `create_pretrain_backbone.py` to extract only the backbone from your checkpoint.

Example:
```bash
python scripts/create_pretrain_backbone.py logdirs/eca_nfnet_l0_Exp_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromXCV2Best_PseudoF2PT05MT01P04I3_MinorOverSampleV1/fold_0/checkpoints/last.ckpt logdirs/eca_nfnet_l0_Exp_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromXCV2Best_PseudoF2PT05MT01P04I3_MinorOverSampleV1/fold_0/checkpoints/last_backbone.ckpt
```

### Inference and Model Compilation

After [training your models](#training), use `main_inference_and_compile.py` to compute metrics and compile the model into OpenVINO format.

```bash
CUDA_VISIBLE_DEVICES="$1" python scripts/main_inference_and_compile.py inference_configs/{your_favourite_config}.py
```

Make sure that the name of your config from `inference_configs` matches the config from `train_configs`.

After running the script, you will get metrics in stdout and 2 new folders in the respective logdir:
- `onnx_ensem_5first_folds` - model converted into ONNX
- `onnx_ensem_5first_folds_openvino_fp16` - model compressed into fp16 and converted into OpenVINO

### Preparing Pseudo Labels

To understand how pseudo labels were prepared, check out the following notebook: [notebooks/create_pseudo.ipynb]

## Cite

```
@article{TODO,
  title={TODO},
  author={Sydorskyi, Volodymyr and Gonçalves, Fernando},
  conference={CLEF 2025},
  pages={TODO},
  year={2025},
  publisher={TODO}
}
```
