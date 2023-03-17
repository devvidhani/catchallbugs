import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

import sys
print(sys.path)

# Import your custom metric from my_metrics.py
from autogluon_custom_metric_serializable import probabilistic_f1_scorer

# Load your dataset
train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

# Prepare the data for AutoGluon
label_column = 'class'
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

# Create a TabularPredictor with your custom metric as eval_metric argument
predictor = TabularPredictor(label=label_column, eval_metric=probabilistic_f1_scorer)
# predictor.fit(train_data, excluded_model_types=['GBM', 'XGB']) #, time_limit=600)  # Set a time limit for training (in seconds)
predictor.fit(train_data, num_gpus=1) #, excluded_model_types=['GBM', 'XGB']) #, time_limit=600)  # Set a time limit for training (in seconds)
# predictor = TabularPredictor(label=label_column).fit(train_data=train_data,
#                                                      eval_metric=probabilistic_f1_scorer)


# Display the leaderboard
leaderboard = predictor.leaderboard(test_data, silent=False)
print(leaderboard)
