import json
import os
import sys

sys.path.append(os.getcwd() + "\\dashboard\\utils")
import config


def get_matrix_providers():
    """
    will open oneyeartransitions.json and read 'providers'
    :return: list of transition matrix providers used in oneyeartransitions.json
    """

    file_loc = config.model_inputs["one_year_matrix_file"]
    f = open(file_loc)
    data_dict = json.load(f)

    try:
        providers = data_dict["providers"]
    except KeyError:
        raise Exception("Parameter Error: Could not find 'providers' in json file")

    return providers


def get_one_year_matrix(provider):
    """
    function that will return the one year transition matrix of the submitted provider
    :param provider: Credit Metrics, Moodys, S&P
    :return: one year transition matrix
    """

    file_loc = config.model_inputs["one_year_matrix_file"]
    f = open(file_loc)
    data_dict = json.load(f)

    try:
        ordered_keys = data_dict["ordered_matrix_keys"]
        matrix = data_dict["data"][provider]["one_year_transitions"]
    except KeyError:
        raise Exception("Data Provider Error: Could not find data on {%s}. See docstrings Provider param for details" % provider)

    return ordered_keys, matrix

