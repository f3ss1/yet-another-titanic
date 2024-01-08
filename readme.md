# Yet another Titanic

## How to run

To **train the model** use:

```bash
python3 train.py
```

Since the project uses [Hydra](https://hydra.cc/), you should be able to
tune the config parameters in line by doing something like:

```bash
python3 train.py random_seed=13
```

To **make the inference** (generate the `submission.csv` file using the trained model) use

```bash
python3 infer.py
```

The project is configured via the `config.yaml` available in `configs` folder.

## Project overview

### Dataset
The classic [Titanic](https://www.kaggle.com/competitions/titanic) dataset.
I used the `train.csv` to train my model and `test.csv` to make predictions on.

### Model
The project uses the [catboost](https://catboost.ai/) gradient boosting model to generate predictions
if the person would survive the catastrophe or not. The predictions are saved to
the `submission.csv` file.

### DVC
The data is managed by [dvc](https://dvc.org/), so it is not directly present in the GitHub
repo. The `dvc pull` is called inside the `train.py` and `infer.py` so there is no
need to do that manually. The data is stored on the Google Drive as backend.

### MLflow logging

The project useds [MLflow](https://mlflow.org/) to log the training process.
