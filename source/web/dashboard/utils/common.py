import json
import os
import sys

from datetime import datetime

sys.path.append(os.getcwd() + "\\dashboard\\utils")
import config


def open_script_log_file(script_name):
    """
    opens a log file for a script
    :param script_name: name of the script that is being run
    :return: file object
    """

    file_creation = str(datetime.now()).replace(":", "_").replace(" ", "_")
    logs_folder = config.settings["script_log_loc"]
    log_file = logs_folder + "\\" + script_name + "_" + file_creation + ".txt"
    f = open(log_file, "w")
    
    return f

def load_matrix_json():
    """
    gets the oneyeartransitions.json and returns the dictionary
    :return:
    """

    file_loc = config.model_inputs["one_year_matrix_file"]
    f = open(file_loc)
    data_dict = json.load(f)

    return data_dict


def get_correlations_list():
    """
    gets the list of built in correlation assumptions
    :return: correlation list
    """

    correlations = config.model_inputs["correlations"]
    return correlations


def get_matrix_providers():
    """
    will open oneyeartransitions.json and read 'providers'
    :return: list of transition matrix providers used in oneyeartransitions.json
    """

    data_dict = load_matrix_json()

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

    data_dict = load_matrix_json()

    try:
        ordered_keys = data_dict["ordered_matrix_keys"]
        matrix = data_dict["data"][provider]["one_year_transitions"]
    except KeyError:
        raise Exception("Data Provider Error: Could not find data on {%s}. See docstrings Provider param for details" % provider)

    return ordered_keys, matrix

