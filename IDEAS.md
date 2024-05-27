# All Ideas

## Low Priority

- 2024 XC metadata - https://www.kaggle.com/datasets/kunihikofurugori/birdclef2024-metadataset
- torch.jit optimization - https://www.kaggle.com/competitions/birdclef-2024/discussion/492649
    ```python
    # Load Your Model
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Magic Line Of Code That Optimizes The Model For Inference
    model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    ```
- Audio augmentations - https://www.kaggle.com/competitions/birdclef-2024/discussion/490922
- Augmentations choice - https://www.kaggle.com/competitions/birdclef-2024/discussion/493131
- Birds denoising - https://www.kaggle.com/code/lihaoweicvch/bird-sound-denoise-by-deep-model
- Data from nearby comps - https://www.kaggle.com/competitions/birdclef-2024/discussion/494516
- Folds variations - https://www.kaggle.com/competitions/birdclef-2024/discussion/494534
- Guy coppied predictions in one .ogg file and another guy fixed it and got worse score !!!
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/492135
- BIRB from Competition hosts !!!
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/491581#2768487
- Data From Last coomp !!!
    - https://zenodo.org/records/10943500
- Hierarchical image classification
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/497566
- Interesting relation between rating and number of secondary labels
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/493605
- Taxonomy comments from hosts
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/491377
- Far away birds are taken into account
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/497830

## High Priority

- Found duplicates - https://www.kaggle.com/competitions/birdclef-2024/discussion/494134
- Check ebird - https://www.kaggle.com/competitions/birdclef-2024/discussion/506449
- Use other birds

# Top ideas

# Checked

## Helped

## Did not help

- Should we treat this task as multiclass - https://www.kaggle.com/competitions/birdclef-2024/discussion/490970#2747028

- Google Bird Model - https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4
    - Comment: Used for picking vocalized regions but LB score decreased
- Use first 5 seconds for training OR Google Bird classifier for finding vocalized chunks !!!
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/497539
- Try submit in ONNX format
- Openvino MultiThread - https://www.kaggle.com/competitions/birdclef-2024/discussion/494665
- Colored and Human noise from audimentations
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/494443
- Take a look at BirdNET
    - https://www.kaggle.com/competitions/birdclef-2024/discussion/496571
