import pandas as pd
from catboost import CatBoostClassifier, Pool

import dvc.api
import mlflow

from yet_another_titanic.preprocessing import DataCleaner, SimplePreprocessor
from yet_another_titanic.callbacks import MLflowLoggingCallback


with dvc.api.open('data/train.csv') as f:
    data = pd.read_csv(f)

cleaner = DataCleaner()
data = cleaner.transform(data)

train_data = data.drop('Survived', axis=1)
train_labels = data[['Survived']]

processor = SimplePreprocessor()
train_data = processor.fit_transform(train_data)

mlflow.start_run()

hyperparams = {'learning_rate': 0.1}
mlflow.log_params(hyperparams)

model = CatBoostClassifier(custom_metric='F1', **hyperparams)
train_pool = Pool(train_data, train_labels)

model.fit(train_pool, callbacks=[MLflowLoggingCallback()])

model.save_model('models/model')
mlflow.end_run()
