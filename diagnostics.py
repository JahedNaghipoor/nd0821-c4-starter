"""
Author: Jahed Naghipoor
Date: December, 2021
This script is used for diagnosis of the data
Pylint score: 9.86/10
"""
import timeit
import os
import json
import logging
import subprocess
import pickle
import pandas as pd
import numpy as np

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_path = config['output_folder_path']
dataset_file = os.path.join(dataset_path, 'finaldata.csv')
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
model_file = os.path.join(output_model_path, 'trainedmodel.pkl')

# Function to get model predictions


def model_predictions(X_test):
    """
    Loads deployed model to predict on data provided
    Args:
        X_test (pandas.DataFrame): Dataframe with features
    Returns:
        y_prediction: Model predictions
    """
    # read the deployed model and a test dataset, calculate predictions
    logging.info("Loading deployed model")
    model = pickle.load(open(model_file, 'rb'))

    logging.info("Running predictions on data")
    y_prediction = model.predict(X_test)
    return y_prediction

# Function to get summary statistics


def dataframe_summary():
    """
    calculate summary statistics for the dataframe

    Returns:
        statistics_dict: dictionary of statistics
    """
    logging.info("Loading and preparing finaldata.csv")

    dataframe = pd.read_csv(dataset_file)
    dataframe_numbers = dataframe.drop(
        ['exited'], axis=1).select_dtypes('number')

    logging.info("Calculating statistics for data")
    statistics_dict = {}
    for column in dataframe_numbers.columns:
        mean = dataframe_numbers[column].mean()
        median = dataframe_numbers[column].median()
        std = dataframe_numbers[column].std()

        statistics_dict[column] = {
            'mean': round(mean, 3),
            'median': round(median, 3),
            'std': round(std, 3)}

    return statistics_dict


def missing_percentage():
    """
    Function to get missing percentage

    Returns:
        missing_list: list of missing values
    """
    logging.info("Loading and preparing finaldata.csv")
    dataframe = pd.read_csv(dataset_file)

    logging.info("Calculating missing data percentage")
    missing_list = {col: {'percentage': percentage} for col, percentage in zip(
        dataframe.columns, dataframe.isna().sum() / dataframe.shape[0] * 100)}

    return missing_list


def execution_time():
    """
    Function to get timings

    Returns:
    timing_list: list of timings
    """
    # timing of ingestion.py
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'],
                       capture_output=True, check=True)
    timing_for_ingestion = timeit.default_timer() - starttime

    # timing of training.py
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'],
                       capture_output=True, check=True)
    timing_for_training = timeit.default_timer() - starttime

    ingestion_time = []
    for _ in range(10):
        ingestion_time.append(timing_for_ingestion)

    training_time = []
    for _ in range(10):
        training_time.append(timing_for_training)

    timing_list = [
        {'ingest_time_mean': round(np.mean(ingestion_time), 3)},
        {'train_time_mean': round(np.mean(training_time), 3)}
    ]
    return timing_list

# Function to check dependencies


def outdated_packages_list():
    """
    Function to check outdated packages

    Returns:
        dep: dependencies of the package
    """
    dep = subprocess.run(
        'pip-outdated requirements.txt',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8').stdout

    return dep


if __name__ == '__main__':
    logging.info("Loading and preparing testdata.csv")

    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    test_df = test_df.drop(['corporation', 'exited'], axis=1)

    print("Model predictions on testdata.csv:",
          model_predictions(test_df))

    print("Missing percentage")
    print(json.dumps(missing_percentage(), indent=4))

    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4))

    print("Execution time")
    print(json.dumps(execution_time(), indent=4))

    print("Outdated Packages")
    print(outdated_packages_list())
