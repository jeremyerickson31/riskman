#  this file serves as an entry point to run a 'script'
#  each 'script' is a function in this file
# 'scripts' may use pieces in the dashboard/utils package

import sys
import json
import random
from dashboard.utils import common, engines, config


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
    providers_list = common.get_matrix_providers()

    common.script_logger(f, "Getting list of correlation assumptions")
    correlations_list = common.get_correlations_list()

    common.script_logger(f, "Getting asset return distribution parameters")
    mu, sigma = common.get_asset_return_distribution_params()

    common.script_logger(f, "Retrieved Providers List %s" % providers_list)
    common.script_logger(f, "Retrieved Correlations List %s" % correlations_list)
    common.script_logger(f, "Retrieved Mu: %s, Sigma: %s" % (mu, sigma))

    common.script_logger(f, "Beginning main calculation loop for each combination of Providers and Correlations")
    for provider in providers_list:

        # make new dictionary to hold results of joint trans probability calculations
        joint_matrices = dict()

        # calculate rating level transition thresholds, dependent only on one year transition matrix and mu and sigma
        common.script_logger(f, "Generating Rating Level Thresholds for: %s" % provider)
        rating_level_thresholds, logs = engines.calc_rating_transition_thresholds(provider, mu, sigma)
        common.script_logger(f, logs)
        common.script_logger(f, "Rating Level Thresholds Complete for %s " % provider)

        # store provider specific rating_level_thresholds for json output
        transition_thresholds = rating_level_thresholds

        common.script_logger(f, "Storing Provider specific rating transition thresholds in json")
        # store provider specific transition thresholds
        thresh_filename = config.model_inputs["transition_thresholds_folder"] + str(provider) + ".json"
        thresh_file = open(thresh_filename, "w")
        json.dump(transition_thresholds, thresh_file, indent=4)
        thresh_file.close()
        common.script_logger(f, "FILENAME: " + str(thresh_filename))

        # get rating_level_threshold keys to loop through for joint probability matrices
        rating_labels = list(transition_thresholds.keys())

        # use the rating level thresholds in numerical integration
        common.script_logger(f, "Begin Bivariate Normal Numerical Integration Between Thresholds")
        for correlation in correlations_list:

            common.script_logger(f, "Calculate Joint Matrices for Provider: %s and Correlation: %s" % (provider, correlation))

            # make the bivariate Gauss correlation matrix
            common.script_logger(f, "Get bivariate Gauss correlation matrix")
            corr_mat = common.make_bivariate_gauss_corr_mat(correlation)
            common.script_logger(f, "Gauss Correlation Matrix: [" + str(corr_mat[0]) + "," + str(corr_mat[1]) + "]")

            # double loop to get all combinations
            for rating_1 in rating_labels:
                for rating_2 in rating_labels:

                    # pair used to denote combo of rating_1 and rating_2 for json storage
                    rating_pair = rating_1 + "|" + rating_2  # ie: AA|CCC, first bond AA second bond CCC

                    common.script_logger(f, "Get Thresholds for Rating 1: %s, Rating 2: %s" % (rating_1, rating_2))
                    rating_1_thresholds = transition_thresholds[rating_1]
                    rating_2_thresholds = transition_thresholds[rating_2]

                    common.script_logger(f, "Submitting Thresholds and Correlation Matrix for numerical integration")
                    matrix, logs = engines.threshold_numerical_integration(rating_1_thresholds, rating_2_thresholds, corr_mat)
                    common.script_logger(f, logs)
                    common.script_logger(f, "Matrix calculation complete. Next Pair of ratings")

                    joint_matrices[rating_pair] = matrix

            common.script_logger(f, "Joint Transition Probabilities Complete for %s and %s " % (provider, correlation))
            common.script_logger(f, "Storing Provider-Correlation specific matrices in json")

            # store provider-correlation specific joint transition probability matrices
            joint_trans_filename = config.model_inputs["joint_matrices_folder"] + str(provider) + "_" + str(correlation) + ".json"
            joint_trans_file = open(joint_trans_filename, "w")
            json.dump(joint_matrices, joint_trans_file, indent=4)
            joint_trans_file.close()
            common.script_logger(f, "FILENAME: " + str(joint_trans_filename))

            common.script_logger(f, "Next Correlation")

        common.script_logger(f, "All Correlations Complete. Next Provider")

    common.script_logger(f, "All Providers Complete.")


def example_three_bond_calculation_analytical():
    """
    This script will test the credit risk calculation on three bonds
    This script runs engines.run_portfolio_credit_risk with run_type = analytical
    This script is meant as a demonstration script as well as a testing script for the
    :return:
    """

    script_name = "example_three_bond_calc_analytic"

    # open log file for script logging
    f = common.open_script_log_file(script_name)
    logger = common.script_logger
    logger(f, "******************")
    logger(f, "Script Log for: " + script_name)
    logger(f, "******************")

    # ###################################################################################
    # #########################      BEGIN USER INPUTS      #############################
    # ###################################################################################
    use_provider = "Credit Metrics"
    use_correlation = 0.30

    # make some fake bonds. fixed rate annuals for now
    bond1_properties = {"bond_name": "bondA",
                        "par": 100, "coupon": 0.06, "maturity": 5, "notional": 4000000.00,
                        "rating": "BBB", "seniority": "Senior Unsecured"}
    bond2_properties = {"bond_name": "bondB",
                        "par": 100, "coupon": 0.05, "maturity": 3, "notional": 2000000.00,
                        "rating": "A", "seniority": "Senior Unsecured"}
    bond3_properties = {"bond_name": "bondC",
                        "par": 100, "coupon": 0.10, "maturity": 2, "notional": 1000000.00,
                        "rating": "CCC", "seniority": "Senior Secured"}

    # ###################################################################################
    # #########################      END OF USER INPUTS      ############################
    # ###################################################################################

    results = engines.run_portfolio_credit_risk([bond1_properties, bond2_properties, bond3_properties], "analytical")
    print(results)


def example_three_bond_calculation_monte():
    """
    This script will test the credit risk monte carlo simulation on two bonds
    This script is meant as a demonstration script as well as a testing script for the
        core engines.py components that are used in the credit risk calculations
    The calculation is the Monte Carlo approach which involves the following:
        (1) Take correlation matrix and generate correlated random numbers with Cholesky Decomposition
        (2) use correlated randoms to determine final rating for each bond
        (3) use rating level forward rates to re-value the bonds and get value for entire portfolio
        (4) do this 10,000 times and get alpha percentile for Credit VaR
    :return:
    """


def make_random_portfolio(num_securities, portfolio_type, to_db=True):
    """
    script that will make a random portfolio and insert into DB / target file
    target is either DB or File name
    :return:
    """
    rating_buckets = {"IG": ["AAA", "AA", "A", "BBB"],
                      "Non-IG": ["BB", "B", "CCC"],
                      "Balanced": ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]}

    portfolio = list()
    for i in range(1, num_securities + 1):
        bond_name = "bond_" + str(i)
        par = 100.00
        maturity = random.randrange(1, 6, 1)
        notional = random.randrange(1, 20, 1) * 1000000.00
        seniority = "Senior Unsecured"
        coupon = None
        rating = None

        # get a rating and get an appropriate coupon given that rating (Junk bonds offer higher coupons)
        if portfolio_type == "IG":
            rating = rating_buckets["IG"][random.randrange(0, len(rating_buckets["IG"]))]
            coupon = random.randrange(1, 6) / 100.00  # IG bonds have lower yields generally
        elif portfolio_type == "Non-IG":
            rating = rating_buckets["Non-IG"][random.randrange(0, len(rating_buckets["Non-IG"]))]
            coupon = random.randrange(6, 15) / 100.00  # Non-IG bonds have higher yields generally
        elif portfolio_type == "Balanced":
            rating = rating_buckets["Balanced"][random.randrange(0, len(rating_buckets["Balanced"]))]
            if rating in rating_buckets["IG"]:
                coupon = random.randrange(1, 6) / 100.00
            elif rating in rating_buckets["Non-IG"]:
                coupon = random.randrange(6, 15) / 100.00

        portfolio.append({"bond_name": bond_name,
                          "par": par, "coupon": coupon, "maturity": maturity, "notional": notional,
                          "rating": rating, "seniority": seniority})

    if to_db is True:
        # open DB connection
        db_conn = common.open_db_connection()
        cursor = db_conn.cursor()

        # delete existing entries
        cursor.execute("DELETE FROM portfolio_risk_management.fixed_income_securities")

        # insert new portfolios
        for bond in portfolio:
            insert_string = "INSERT INTO portfolio_risk_management.fixed_income_securities " \
                            "(id, name, par, coupon, maturity, notional, rating, seniority, portfolio_name) " \
                            "VALUES " \
                            "(%s, '%s', %s, %s, %s, %s, '%s', '%s', '%s')" \
                            % ('NULL', bond['bond_name'], bond['par'], bond['coupon'], bond['maturity'],
                               bond['notional'], bond['rating'], bond['seniority'], portfolio_type)
            cursor.execute(insert_string)
            db_conn.commit()

    else:
        # dump to a json file
        folder = config.model_inputs["sample_portfolios_folder"]
        file_name = folder + "\\" + portfolio_type + "_" + str(num_securities) + ".json"
        f = open(file_name, "w")
        json.dump({"portfolio_bonds": portfolio}, f, indent=4)
        f.close()

if __name__ == "__main__":
    #example_three_bond_calculation_analytical()
    make_random_portfolio(10, "IG", False)

