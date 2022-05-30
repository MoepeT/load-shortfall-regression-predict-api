"""
    Simple Script to test the API once deployed
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------
    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.
"""

# Import dependencies
import requests
import pandas as pd
import numpy as np

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Kaggle challenge.
test = pd.read_csv('./data/df_test.csv')


# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://63.32.46.90:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)







# Display the prediction result 
import requests
import re
import json
from datetime import datetime

quarterdateformat = '%Y-%m-%d'


def camelize(string):
    return "".join(string.split(" "))

def convert_types(d):
    for k, v in d.items():
        #print(k, type(v))
        new_v = v
        if type(v) is str:
            #match for float
            if re.match('[-+]?[0-9]*\.[0-9]+', v):  
                new_v = float(v)

            #match for date
            if re.match('([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))', v):  
                new_v = datetime.strptime(v, quarterdateformat).date()


        d[k] = new_v
    d = {camelize(k): v for k, v in d.items()}
    return d

url = 'http://54.72.214.68:5000/api_v0.1'
params = {'datatyupe' : 'json'}
r = requests.get(url, params)
    return data



# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://3.251.90.89:5000/api_v0.1'
url = 'http://54.72.214.68:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result 
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()[0]}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
        

