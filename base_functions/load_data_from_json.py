import json
import os.path

"""
load_data_from_json(json_path, verbose):
This function loads the content of a json-file into an object and returns it.
INPUT:
    json_path (String): The path to the json-file.
    verbose (Boolean, False): If true, the status of the operations are printed
OUTPUT:
    data (dict): The content of the json-file
"""


def load_data_from_json(json_path, verbose=False):

    assert os.path.isfile(json_path), f"No file found at {json_path}"

    if verbose:
        print(f"Load json from {json_path}")

    with open(json_path) as file:
        data = json.load(file)

    return data
