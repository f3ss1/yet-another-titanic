from pathlib import Path

import dvc.api
import hydra
import mlflow
import pandas as pd
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig

from yet_another_titanic.callbacks import MLflowLoggingCallback
from yet_another_titanic.preprocessing import Pipeline
from yet_another_titanic.utils import create_parents, get_git_commit_hash, seed_everything


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    if config["use_mlflow"]:
        mlflow.set_tracking_uri(config["mlflow_uri"])

    seed_everything(config["random_seed"])

    with dvc.api.open(config["data"]["train_path"]) as file:
        data = pd.read_csv(file)

    target = config["data"]["target"]

    transform_pipeline = Pipeline(
        config["data"]["columns_to_drop"], config["data"]["columns_to_mix"], config["data"]["mixed_column_name"]
    )
    train_data = transform_pipeline.transform(data)

    train_labels = train_data[[target]]
    train_data = train_data.drop(target, axis=1)
    hyperparams = config["model_params"]

    if config["use_mlflow"]:
        mlflow.start_run()
        git_hash = get_git_commit_hash()
        if git_hash is not None:
            mlflow.log_param("git_commit", git_hash)

        mlflow.log_params(hyperparams)

    model = CatBoostClassifier(custom_metric="F1", **hyperparams)
    train_pool = Pool(train_data, train_labels, cat_features=config["data"]["cat_features"])

    if config["use_mlflow"]:
        model.fit(train_pool, callbacks=[MLflowLoggingCallback()])
    else:
        model.fit(train_pool)

    model_path = Path("models/model")
    create_parents(model_path)

    model.save_model(model_path)

    if config["use_mlflow"]:
        mlflow.end_run()


if __name__ == "__main__":
    main()
