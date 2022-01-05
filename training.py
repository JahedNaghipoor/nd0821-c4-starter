"""
Author: Jahed Naghipoor
Date: December, 2021
This script is used for training the model on the ingested data
Pylint score: 9.57/10
"""
import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_path = config['output_folder_path']
output_model_path = config['output_model_path']


# Function for training the model
def train_model(dataframe):
    """
    Train ingested data using  logistic regression model and save the model
    """

    label = dataframe["exited"]
    features = dataframe.drop(["exited", "corporation"], axis=1)

    X_train, _, y_train, _ = train_test_split(
        features, label, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # use this logistic regression for training
    logistic_reg = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0)

    # fit the logistic regression to your data
    model = logistic_reg.fit(X_train, y_train)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)

    model_file = os.path.join(output_model_path, 'trainedmodel.pkl')
    pickle.dump(model, open(model_file, 'wb'))


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(dataset_path, 'finaldata.csv'))
    train_model(df)
