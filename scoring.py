"""
Author: Jahed Naghipoor
Date: December, 2021
This script used for scoring the model on the test data
Pylint score: 10/10
"""

import os
import json
import pickle
import pandas as pd
from sklearn.metrics import f1_score


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = os.path.join(
    config['output_model_path'],
    'trainedmodel.pkl')


# Function for model scoring
def score_model():
    """
    This function loads a trained model and the test data, and calculate an F1 score
    for the model on the test data and saves the result to the latestscore.txt file
    """

    # load the model
    model = pickle.load(open(output_model_path, 'rb'))

    # read test data
    test_data_file = os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(test_data_file)

    # calculate an F1 score for the model relative to the test data
    y_true = test_data["exited"]
    test_data.drop(["exited", "corporation"], axis=1, inplace=True)
    y_pred = model.predict(test_data)
    f1 = f1_score(y_true, y_pred)

    # write the result to the latestscore.txt file
    score_file = os.path.join(config['output_folder_path'], 'latestscore.txt')
    with open(score_file, 'w') as file:
        file.write(str(f1))
        
    return str(f1)    


if __name__ == '__main__':
    score_model()
