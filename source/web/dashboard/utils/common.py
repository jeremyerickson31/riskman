import json
import os
import sys
import numpy
import scipy.stats

from io import TextIOWrapper
from datetime import datetime
import mysql.connector

sys.path.append(os.getcwd() + "\\dashboard\\utils")
import config


def open_db_connection():
    """
    will open connection to portfolio_risk_management database with user = riskmanuser
    :return:
    """

    try:
        connection = mysql.connector.connect(host=config.db_conn['local_host'],
                                             user=config.db_conn['username'],
                                             passwd=config.db_conn['password'],
                                             database=config.db_conn['database'])
    except:
        raise Exception("DB Connection Error: Couldn't connect to DB. Check username, password")

    return connection


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
    f.close()

    return data_dict


def load_transition_thresholds(provider):
    """
    gets the transition thresholds for the specified provider
    :return:
    """

    file_loc = config.model_inputs["transition_thresholds_folder"] + provider + ".json"
    f = open(file_loc)
    trans_thresholds = json.load(f)
    f.close()

    return trans_thresholds


def load_bond_portfolio_json(portfolio_name):
    """
    will load sample portfolios
    :param portfolio_name: name of portfolio to load
    :return:
    """

    portfolio_folder = config.model_inputs["sample_portfolios_folder"]
    portfolio_files = os.listdir(portfolio_folder)
    for portfolio_file in portfolio_files:
        name = portfolio_file.split(".")[0]
        if name == portfolio_name:
            f = open(portfolio_folder + "\\" + portfolio_file)
            portfolio_data = json.load(f)
            bond_list = portfolio_data["portfolio_bonds"]  # this line was updated
            f.close()
            break
    else:
        bond_list = {"NONE"}

    return bond_list


def get_available_bond_portfolios():
    """
    will grab all the file names in the sample_po
    :return:
    """

    files = os.listdir(config.model_inputs["sample_portfolios_folder"])
    file_names = [name.split(".")[0] for name in files]

    return file_names


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
    f.close()

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


def get_price_decimals():
    """
    fetches the rounding digits from config
    :return: number of decimals to carry, as int
    """

    decimals = config.model_inputs["price_decimals"]

    return decimals


def fmt_num(number, fmt, power, scale):
    """
    small function to format numbers
    :param number: the number to be formatted
    :param fmt: type of formatting (dollar, percent later)
    :param power: optional, power=1 for means, power=2 for variances
    :param scale: optional, Millions, Billions etc
    :return: formatted number
    """

    formats = ['$']
    if fmt not in formats:
        raise Exception("Parameter Error: unsupported number format. Currently supports " + str(formats))

    scales = ['MM']
    if scale not in scales:
        raise Exception("Parameter Error: unsupported number scale. Currently supports " + str(scales))

    if fmt == "$":
        if power == 1:
            if scale == "MM":
                formatted = number / 1000000.0
            if scale == "BB":
                formatted = number / 1000000000.0

        if power == 2:
            if scale == "MM":
                formatted = number / (1000000.0 ** 2)
            if scale == "BB":
                formatted = number / (1000000000.0 ** 2)

    return formatted


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


def get_interest_rate_curves():
    """
    fetch interest rate curves found in the param folder
    :return:
    """

    file_loc = config.model_inputs["interest_rate_curves"]
    f = open(file_loc)
    data_dict = json.load(f)
    rating_level_rates_dict = data_dict["rates"]
    f.close()

    return rating_level_rates_dict


def get_recovery_in_default(seniority):
    """
    function will look up the recovery value in the event of default
    :param seniority: the seniority of the bond to lookup
    :return: recovery, as float
    """

    file_loc = config.model_inputs["recoveries_in_default"]
    f = open(file_loc)
    data_dict = json.load(f)
    f.close()

    try:
        recovery = data_dict[seniority]
    except KeyError:
        raise Exception("KeyError: Recovery JSON has no seniority level {%s}" % seniority)

    return recovery


def rand_to_rating(initial_rating, provider, rand_num):
    """
    this function accepts a random number between 0 and 1, applies the normal inverse function, and finds what rating
    that corresponds to for a given provider transition thresholds
    :param initial_rating: initial rating
    :param rand_num: float between 0 and 1
    :param provider: Credit Metrics, Moodys, SP
    :return: rating
    """

    if not isinstance(rand_num, float):
        raise Exception("Parameter Error: rand_num must be a float. Got " + str(type(rand_num)))
    if not 0.0 < rand_num < 1.0:
        raise Exception("Value Error: rand_num must be float between 0.0 and 1.0")

    # load transition thresholds
    trans_thresholds = load_transition_thresholds(provider)
    initial_rating_thresholds = trans_thresholds[initial_rating]

    # get ordered ratings so we can go from D --> AAA
    ratings = get_ordered_rating_keys()
    rating_keys = list(ratings.keys())
    rating_keys.sort()
    rating_keys.reverse()  # results in [8,7,6,...] one nunber for each major rating level in oneyeartransitions.json)

    # convert random number between 0 and 1 to a standard normal value
    transition_value = scipy.stats.norm(0., 1.0).ppf(rand_num)

    for key in rating_keys:
        rating_level = ratings[key]
        threshold = initial_rating_thresholds[rating_level]

        if transition_value < threshold:
            final_rating = rating_level
            break
        else:
            continue
    else:
        # if the less than is never hit then the rand number is really high and the highest rating is the result
        final_rating = ratings(rating_keys[1])

    return final_rating


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


def run_bond_properties_check(bond_props):
    """
    function will validate that all the required bond properties are present
    :param bond_prop: accepts a list of dictionaries with bond properties
    :return: ist of wrong stuff, if no wrong stuff then empty list
    """

    if not isinstance(bond_props, list):
        raise Exception("Parameter Error: bond_props must be a list of dictionaries")
    for entry in bond_props:
        if not isinstance(entry, dict):
            raise Exception("Parameter Error: each entry in bond_props must be a dictionary")

    # list of required properties
    req_props = set(config.model_inputs["required_bond_properties"])

    incomplete_properties = list()
    for properties in bond_props:
        props = set(properties.keys())

        prop_diff = req_props - props
        if prop_diff != set():
            incomplete_properties.append({"properties": properties, "missing": prop_diff})

    return incomplete_properties


if __name__ == "__main__":
    out = fmt_num(14308956522.32637, '$', 2, 'MM')
    print(out)
    """
    conn = open_db_connection()
    curr = conn.cursor()
    curr.execute("SHOW DATABASES;")
    print(curr.fetchall())
    """