#  this file serves as an entry point to run a 'script'
#  each 'script' is a function in this file
# 'scripts' may use pieces in the dashboard/utils package

import sys
from datetime import datetime
from dashboard.utils import common, engines


def build_joint_trans_probs():
    """
    This function will generate the joint rating transition probabilities using the approach outlined
    in the Credit Metrics Technical Document

    The joint rating transition probabilities are dependent on the one year migration matrix, the mean and
    standard deviation of the assumed asset return distribution (assumed 0,1 in the standard normal), as well
    as the correlation of the two normal variables

    This script makes use of two functions found in engines.py
    (1) calc_rating_transition_thresholds()
        This function calculates thr rating transition thresholds. This is dependent only on the one year transition
        matrix and the asset return distribution parameters. These are generally static and there is no option to change
        these in the dashboard by the user
    (2) threshold_numerical_integration()
        this function takes the rating transition thresholds for two credit exposures and calculates the joint
        transition probability matrix. This matrix is dependent on the thresholds (which is dependent on the one year
        transition matrix) as well as the correlation parameter. In this script the correlations are set and so the
        joint transition matrices are also set, but the function allows for a user defined correlation in order to
        calculate the joint matrix on the fly.

    The joint rating transition probabilities are specified for each combination of transition matrix provider
    and correlation. results are stored in json for quick lookup in the dashboard
    :return: None , but results are stored in JSON
    """

    script_name = "build_joint_trans_probs"
    transition_thresholds = dict()
    joint_matrices = dict()

    # open log file for script logging
    f = common.open_script_log_file(script_name)
    common.script_logger(f, "******************")
    common.script_logger(f, "Script Log for: " + script_name)
    common.script_logger(f, "******************")

    # load list of transition matrix providers, correlation assumptions and asset return distribution params
    common.script_logger(f, "Getting list of transition matrix providers")
    providers_list = ["SP Ratings"]  # common.get_matrix_providers()

    common.script_logger(f, "Getting list of correlation assumptions")
    correlations_list = [0.15]  # common.get_correlations_list()

    common.script_logger(f, "Getting asset return distribution parameters")
    mu, sigma = common.get_asset_return_distribution_params()

    common.script_logger(f, "Retrieved Providers List %s" % providers_list)
    common.script_logger(f, "Retrieved Correlations List %s" % correlations_list)
    common.script_logger(f, "Retrieved Mu: %s, Sigma: %s" % (mu, sigma))

    common.script_logger(f, "Beginning main calculation loop for each combination of Providers and Correlations")
    for provider in providers_list:

        # make new dictionaries to hold results of threshold and joint trans probs caluclations
        transition_thresholds[provider] = dict()
        joint_matrices[provider] = dict()

        # calculate rating level transition thresholds, dependent only on one year transition matrix and mu and sigma
        common.script_logger(f, "Generating Rating Level Thresholds for: %s" % provider)
        rating_level_thresholds, logs = engines.calc_rating_transition_thresholds(provider, mu, sigma)
        common.script_logger(f, logs)
        common.script_logger(f, "Rating Level Thresholds Complete for %s " % provider)

        # store provider specific rating_level_thresholds for json output
        transition_thresholds[provider] = rating_level_thresholds

        # get rating_level_threshold keys to loop through for joint probability matrices
        rating_labels = list(transition_thresholds[provider].keys())

        # use the rating level thresholds in numerical integration
        common.script_logger(f, "Begin Bivariate Normal Numerical Integration Between Thresholds")
        for correlation in correlations_list:

            # joint matrices are specific to the provider and the correlation
            joint_matrices[provider][correlation] = dict()

            common.script_logger(f, "Calculate Joint Matrices for Provider: %s and Correlation: %s" % (provider, correlation))
            # double loop to get all combinations
            for rating_1 in rating_labels:
                for rating_2 in rating_labels:

                    # pair used to denote combo of rating_1 and rating_2 for json storage
                    rating_pair = rating_1 + "|" + rating_2

                    common.script_logger(f, "Get Thresholds for Rating 1: %s, Rating 2: %s" % (rating_1, rating_2))
                    rating_1_thresholds = transition_thresholds[provider][rating_1]
                    rating_2_thresholds = transition_thresholds[provider][rating_2]

                    common.script_logger(f, "Submitting Thresholds and Correlation for numerical integration")
                    #matrix = engines.threshold_numerical_integration(rating_1_thresholds, rating_2_thresholds, correlation)
                    common.script_logger(f, "Matrix calculation complete. Next Pair of ratings")

                    joint_matrices[provider][correlation][rating_pair] = 'matrix'
            common.script_logger("All Rating Pair Joint Transition Probability Matrices Complete")

    common.script_logger(f, "output json thresholds:")
    common.script_logger(f, str(transition_thresholds))

    common.script_logger(f, "output json joint matrices")
    for provider in providers_list:
        for correlation in correlations_list:
            common.script_logger(f, str(joint_matrices[provider][correlation]))


        # todo store the rating transition thresholds as separate file {'Credit Metrics': {}, 'Moodys': {} }
        # todo store the rating transition probability matrices as separate file {'Credit Metrics': {0.15:{}, 0.2{} }


if __name__ == "__main__":
    supported_scripts = ["build_joint_trans_probs"]

    # inputs = sys.argv()
    inputs = ["", "build_joint_trans_probs"]
    script_to_run = inputs[1]

    if script_to_run == supported_scripts[0]:
        build_joint_trans_probs()

