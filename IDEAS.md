- 2024 XC metadata - https://www.kaggle.com/datasets/kunihikofurugori/birdclef2024-metadataset
- torch.jit optimization - https://www.kaggle.com/competitions/birdclef-2024/discussion/492649
    ```python
    # Load Your Model
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Magic Line Of Code That Optimizes The Model For Inference
    model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    ```
- Should we treat this task as multiclass - https://www.kaggle.com/competitions/birdclef-2024/discussion/490970#2747028 
- Audio augmentations - https://www.kaggle.com/competitions/birdclef-2024/discussion/490922
- Augmentations choice - https://www.kaggle.com/competitions/birdclef-2024/discussion/493131
