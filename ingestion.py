"""
Author: Jahed Naghipoor
Date: December, 2021
This script is used to ingest the data. For that it merges multiple dataframes into one dataframe.
Pylint score: 10/10
"""

import os
import json
import pandas as pd

# Load config.json and get input and output paths
with open('config.json', 'r') as file:
    config = json.load(file)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Function for data ingestion


def merge_multiple_dataframe():
    """
    merge_multiple_dataframe [summary]

    This function merges multiple dataframes into one dataframe.
    For that, it will read all the files in the input folder, then merge them into one dataframe.
    It drops the duplicates and saves the dataframe into the output file finaldata.csv.
    It also saves a record of the files that were ingested into the output file ingestedfiles.txt.
    """

    # Step 1: search for all csv files in input_folder_path
    files = [file for file in os.listdir(
        input_folder_path) if file.endswith('.csv')]

    # Step 2: merge files into  one dataframe
    dataframe = pd.concat([pd.read_csv(os.path.join(input_folder_path, file))
                           for file in files], ignore_index=True)

    # Step 3: de-dupe for dataframe
    dataframe.drop_duplicates(
        subset=['corporation'],
        keep='first',
        inplace=True)

    # Step 4: save df to file
    # create output folder, if does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    # save file as csv in output folder
    output_file = os.path.join(output_folder_path, 'finaldata.csv')
    dataframe.to_csv(output_file, index=False)

    # Step 5: save a record of files read from input folder to output
    # ingestedfiles.txt
    temp_file = None
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as temp_file:
        for item in files:
            temp_file.write(f"{item}\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
