from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

import dvc.api

import hydra
from omegaconf import DictConfig

from yet_another_titanic.utils import seed_everything


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig):
    seed_everything(config['random_seed'])

    with dvc.api.open(config['data']['test_path']) as f:
        data = pd.read_csv(f)

    # Model load placeholder
    transform_pipeline = ...
    model_path = Path('models/model')
    model = CatBoostClassifier().load_model(model_path)
    transformed_data = transform_pipeline.transform(data, clean=False)

    predictions = model.predict(transformed_data)
    result = pd.DataFrame(
        {
            'PassengerId': data['PassengerId'],
            config['data']['target']: predictions
        }
    )
    submit_path = Path(config['predictions_save_path'])
    result.to_csv(submit_path)
