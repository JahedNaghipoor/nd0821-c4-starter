"""
Author: Jahed Naghipoor
Date: December, 2021
This script is used to store files in the deployment directory
Pylint score: 10/10
"""
import os
import shutil
import json

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']


def store_files():
    """
    copy the latestscore.txt value into the deployment directory
    copy the latest model file into the deployment directory
    copy the ingestfiles.txt file into the deployment directory
    """
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)
    # copy the latestscore.txt value into the deployment directory
    shutil.copy(os.path.join(dataset_path,
                'ingestedfiles.txt'), prod_deployment_path)

    # copy the latest model file into the deployment directory
    shutil.copy(os.path.join(output_model_path,
                'trainedmodel.pkl'), prod_deployment_path)

    # copy the ingestfiles.txt file into the deployment directory
    shutil.copy(os.path.join(dataset_path,
                'latestscore.txt'), prod_deployment_path)


if __name__ == '__main__':
    store_files()
