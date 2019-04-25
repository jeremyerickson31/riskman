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

    The joint rating transition probabilities are specified for each combination of transition matrix provider
    and correlation. results are stored in json for quick lookup in the dashboard
    :return: None , but results are stored in JSON
    """

    script_name = "build_joint_trans_probs"
    results = dict()

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

    common.script_logger(f, "Getting asset return ditribution parameters")
    mu, sigma = common.get_asset_return_distribution_params()

    common.script_logger(f, "Retrieved Providers List %s" % providers_list)
    common.script_logger(f, "Retrieved Correlations List %s" % correlations_list)
    common.script_logger(f, "Retrieved Mu: %s, Sigma: %s" % (mu, sigma))

    common.script_logger(f, "Beginning main calculation loop for each combination of Providers and Correlations")
    for provider in providers_list:

        results[provider] = dict()  # new dictionary to store the provider specific results

        # calculate rating level transition thresholds, dependent only on one year transition matrix and mu and sigma
        common.script_logger(f, "Generating Rating Level Thresholds for: %s" % provider)
        rating_level_thresholds, logs = engines.calc_rating_transition_thresholds(provider, mu, sigma)
        common.script_logger(f, logs)
        common.script_logger(f, "Rating Level Thresholds Complete for %s " % provider)

        # use the rating level thresholds in numerical integration
        common.script_logger(f, "Begin Bivariate Normal Numerical Integration Between Thresholds")
        for correlation in correlations_list:
            # run bivariate normal numerical integration between thresholds
            pass

        results[provider]["thresholds"] = rating_level_thresholds
        results[provider]["joint_trans_probs"] = None


if __name__ == "__main__":
    supported_scripts = ["build_joint_trans_probs"]

    # inputs = sys.argv()
    inputs = ["", "build_joint_trans_probs"]
    script_to_run = inputs[1]

    if script_to_run == supported_scripts[0]:
        build_joint_trans_probs()

