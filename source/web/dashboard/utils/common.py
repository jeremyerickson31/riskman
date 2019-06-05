import json
import os
import sys
import numpy

from io import TextIOWrapper
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


def script_logger(file_obj, text):
    """
    function used by the scripts to log certain things to the file as well as print them to console
    :param file_obj: requires an open file object
    :param text: the text or list of texts to log and print
    :return: True / False for success or fail
    """

    if not isinstance(file_obj, TextIOWrapper):
        raise Exception("Parameter Error: first parameter is supposed to be an open file object")

    if not isinstance(text, str) and not isinstance(text, list):
        raise Exception("Parameter Error: second parameter is supposed to be a string or list of strings")

    # print to file and print to console
    if isinstance(text, str):
        text_list = [text]
    else:
        text_list = text

    result = False
    for entry in text_list:
        try:
            timestamp = str(datetime.now()).replace(":", "_").replace(" ", "_")
            log_text = timestamp + ": " + entry
            file_obj.write(log_text + "\n")
            file_obj.flush()
            print(log_text)
        except:
            break
    else:
        result = True

    return result


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


def get_transition_probs(provider, rating):
    """
    function to get a single row from a transition matrix
    :param provider: Credit Metrics, Moodys etc
    :param rating: AAA, AA, A, BBB ...
    :return: dictionary with transition probabilites
    """

    keys, matrix = get_one_year_matrix(provider)
    probabilities = matrix[rating]
    return probabilities


def get_provider_correlation_joint_probs(provider, correlation):
    """
    this function will fetch the pre-calculated joint transition probabilities for the provider-correlation pair
    :param provider: Credit Metrics, SP Ratings, Moodys
    :param correlation: 0.1, 0.2, ....
    :return: dictionary
    """

    joint_probs_loc = config.model_inputs["joint_matrices_folder"]
    file_name = joint_probs_loc + provider + "_" + str(correlation) + ".json"

    f = open(file_name)
    joint_probs_dict = json.load(f)

    return joint_probs_dict


def get_ordered_rating_keys():
    """
    function that will return the ordered keys located in the oneyeartransition.json file
    :return: ordered keys
    """

    data_dict = load_matrix_json()
    ordered_keys = data_dict["ordered_matrix_keys"]

    return ordered_keys


def get_asset_return_distribution_params():
    """
    function to fetch parameters of teh asset return distribution
    :return: mu, sigma tuple
    """

    mu = config.model_inputs["asset_return_distribution"]["mu"]
    sigma = config.model_inputs["asset_return_distribution"]["sigma"]

    return mu, sigma


def get_matrix_sum_tolerance():
    """
    fetches the matrix sum = 1 tolerance from config
    :return: tolerance float
    """

    tol = config.model_inputs["matrix_sum_tolerance"]

    return tol


def make_bivariate_gauss_corr_mat(correlation):
    """
    function that makes the correlation matrix for a bivariate gaussian
    :param correlation: float
    :return: 2x2 numpy array
    """

    # check correlation is float
    if not isinstance(correlation, float):
        raise Exception("Parameter Error: correlation must be a float. Got %s" % str(type(correlation)))

    mu, sigma = get_asset_return_distribution_params()
    matrix = numpy.array([[sigma ** 2, correlation * sigma ** 2], [correlation * sigma ** 2, sigma ** 2]])

    return matrix


def make_bivariate_gauss_mu_mat():
    """
    function that makes the correlation matrix for a bivariate gaussian
    :return: 2x2 numpy array
    """

    mu, sigma = get_asset_return_distribution_params()
    matrix = numpy.array([mu, mu])

    return matrix


def get_recovery_in_default(seniority):
    """
    function will look up the recovery value in the event of default
    :param seniority: the seniority of the bond to lookup
    :return: recovery, as float
    """

    file_loc = config.model_inputs["recoveries_in_default"]
    f = open(file_loc)
    data_dict = json.load(f)

    try:
        recovery = data_dict[seniority]
    except KeyError:
        raise Exception("KeyError: Recovery JSON has no seniority level {%s}" % seniority)

    return recovery


def get_two_asset_combinations(name_list):
    """
    this function will determine the distinct list of two asset sub portfolios from the list of asset names
    :param name_list: list of asset names
    :return: list of two asset sub portfolio combinations
    """

    # check that name_list is a list
    if not isinstance(name_list, list):
        raise Exception("Paramter Error: name_list parameter must be a list. Got %s instead" % str(type(name_list)))

    # check to see that name_list has no duplicates
    if len(set(name_list)) == len(name_list):
        pass
    else:
        raise Exception("Uniquness Error: name_list cannot have duplicate entries. Each entry must be unique")

    combos = list()
    # double loop through names in name_list
    for first in name_list:
        for second in name_list:

            forward_combo_name = first + "-" + second
            backward_combo_name = second + "-" + first

            # don't keep (name_1 - name_1) combo
            if first == second:
                continue

            # only keep one of (name_1-name_2) or (name_2-name_1)
            if (forward_combo_name in combos) or (backward_combo_name in combos):
                continue

            combos.append(forward_combo_name)

    return combos
