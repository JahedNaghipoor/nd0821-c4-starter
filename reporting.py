"""
Author: Jahed Naghipoor
Date: December, 2021
This script used to generate different reports
Pylint score: 10/10
"""

import pickle
import json
import os
import sys
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

import diagnostics

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def plot_confusion_matrix():
    """
    Calculate a confusion matrix using the test data and the deployed model
    """
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    y_true = test_df.pop('exited')
    X_test = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting test data")
    y_pred = diagnostics.model_predictions(X_test)

    logging.info("Plotting and saving confusion matrix")
    ax= plt.subplot()
    fig = sns.heatmap(metrics.confusion_matrix(y_true,y_pred), annot=True, fmt="d")

    ax.set_title("Model Confusion Matrix")
    fig.figure.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))


def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    # load model from the workspace
    model = pickle.load(open(model_path, 'rb'))

    # load test data from csv
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    # calculate confusion matrix using the test data and the deployed model
    y_pred = model.predict(test_data.drop(["exited", "corporation"], axis=1))
    cm = metrics.confusion_matrix(test_data["exited"], y_pred)

    # write the confusion matrix to the workspace
    temp_file = None
    with open(os.path.join(output_model_path, 'confusion_matrix.txt'), 'w') as temp_file:
        temp_file.write(str(cm))


if __name__ == '__main__':
    plot_confusion_matrix()
    score_model()
