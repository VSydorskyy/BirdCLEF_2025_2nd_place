# BirdCLEF+ 2025 2nd place solution

# Pre-requirements

- Ubuntu 22.04.3 LTS x86_64
- CUDA Version: 12.4
- CUDA Driver Version: 550.144.03
- Poetry version 2.1.3
- NVIDIA GeForce RTX 4090 (or any other GPU with VRAM >= 10Gb)
- Hard Disk: 600 Gb. You may need more than 1 Tb in case if you want to pre-train
- RAM: 126 Gb

# Environment

Setup Kaggle credentials in order to download data

```bash
export KAGGLE_USERNAME={KAGGLE_USERNAME}
export KAGGLE_KEY={KAGGLE_KEY}
```

## Setup Poetry

1. [Install Poetry](https://python-poetry.org/docs/#installation)
  - The easiest way is to use [Official Installer guide](https://python-poetry.org/docs/#installing-with-the-official-installer).
  - Pay attention to `poetry --version`. in order to install correct version do : `curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3`. If you have already installed another version - simply change it with next command: `poetry self update 2.1.3`
2. Configure Poetry to create Env in local folder - `poetry config virtualenvs.in-project true`
3. `eval $(poetry env activate)`
4. `poetry install`
5. In order to deactivate env - `exit`
6. In order to remove env - `rm .venv -rf`

# Execute all pipeline

```bash
bash rock_that_bird.sh "{GPU_TO_USE}" # By default: bash rock_that_bird.sh "0"
```

# Inference

- [Kaggle Best Ensemble Inference](https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-ensemble-v2-final-final?scriptVersionId=244942051)
- Kaggle Solo Model Inference

# Solution Description

- [Kaggle Discussion](https://www.kaggle.com/competitions/birdclef-2025/discussion/583699)
- Paper TODO
