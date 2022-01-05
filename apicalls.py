"""
Author: Jahed Naghipoor
Date: December, 2021
This script is used to call the Flask API
Pylint score: 10/10
"""
import os
import sys
import logging
import json
import requests

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Specify a URL
URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = config['test_data_path']
prediction_file = test_data_path,'testdata.csv'
output_model_path = config['output_model_path']

logging.info("Generating report text file")
report_file = os.path.join(output_model_path, 'apireturns.txt')
with open(report_file, 'w') as file:
    file.write('Ingested Data: \n')
    file.write('Statistics Summary\n')
    file.write(requests.get(f'{URL}/summarystats').text)
    file.write('Diagnostics Summary\n')
    file.write(requests.get(f'{URL}/diagnostics').text)
    file.write('Test Data: \n')
    file.write('Model Predictions\n')
    file.write(requests.post(f'{URL}/prediction' \
        ,json={'prediction_file_path': prediction_file}).text)
    file.write('Model Score: '+ requests.get(f'{URL}/scoring').text)
