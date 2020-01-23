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
    This script is meant as a demonstration script as well as a testing script for the
        core engines.py components that are used in the credit risk calculations
    The calculation is the Analytical approach which involves the following:
        (1) for each bond, calculate the value in each possible state
        (2) apply probabilities for each state to get a mean and variance for each bond
        (3) identify each pair of bonds and calculate pair-wise sub-portfolio value in each possible state
        (4) apply joint probabilities to get sub-portfolio mean and variance
        (5) combine individual mean values to get portfolio mean value
        (6) combine pair-wise variances with individual variances to get portfolio variance
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
    use_provider = "SP Ratings"
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

    # some sample rating level forward rates for repricing
    forward_rates = {"AAA": [3.60, 4.17, 4.73, 5.12, 5.83, 6.05, 6.27, 6.68, 7.12],
                     "AA": [3.65, 4.22, 4.78, 5.17, 5.92, 6.12, 6.45, 7.01, 7.38],
                     "A": [3.72, 4.32, 4.93, 5.32, 6.15, 6.29, 6.78, 7.25, 7.50],
                     "BBB": [4.10, 4.67, 5.25, 5.63, 6.37, 6.45, 6.98, 7.37, 7.81],
                     "BB": [5.55, 6.02, 6.78, 7.27, 7.98, 8.15, 8.64, 9.12, 9.53],
                     "B": [6.05, 7.02, 8.03, 8.52, 9.04, 9.73, 10.15, 10.64, 11.30],
                     "CCC": [15.05, 15.02, 14.03, 13.52, 13.07, 12.63, 12.12, 11.70]
                     }

    # ###################################################################################
    # #########################      END OF USER INPUTS      ############################
    # ###################################################################################

    # get master file of joint probabilities for provider/correlation pair
    joint_probs_master = common.get_provider_correlation_joint_probs(use_provider, use_correlation)

    # add bond properties to list for nice looping through bonds
    bond_list = [bond1_properties, bond2_properties, bond3_properties]
    bond_names = [item["bond_name"] for item in bond_list]  # list of names to use for making two asset sub portfolios

    # begin the individual bond calculations
    bond_calcs = dict()  # dictionary for holding the single asset and two asset calculations that have been done
    for bond_properties in bond_list:
        bond = engines.Bond(bond_properties)  # initialize the bond object
        bond.get_transition_probabilities(use_provider)  # fetch transition probs for given provider and self.rating
        bond.calc_prices_under_forwards(forward_rates)  # use provided forward rates to do re-pricing
        bond.calc_price_stats()  # apply transition probabilities to get mean and variance

        logger(f, "----------------------")
        logger(f, bond.name)
        logger(f, "Mean: " + str(round(common.fmt_num(bond.price_stats_dollar["mean"], '$', 1, 'MM'), 3)))
        logger(f, "Var: " + str(round(common.fmt_num(bond.price_stats_dollar["variance"], '$', 2, 'MM'), 3)))
        # common.script_logger(f, "transition probs - " + str(bond.transition_probs))
        # common.script_logger(f, "bond price pct - " + str(bond.rating_level_prices_pct))
        # common.script_logger(f, "bond price dollar - " + str(bond.rating_level_prices_dollar))
        # common.script_logger(f, "price stats pct - " + str(bond.price_stats_pct))
        # common.script_logger(f, "price stats dollar - " + str(bond.price_stats_dollar))

        # add this bond object to the dictionary of bond objects that we have done calculations for
        bond_calcs[bond.name] = {"type": "single_asset", "stats": None, "object": bond}

    # begin the two bond sub portfolio calculations
    logger(f, "-------------------------")
    logger(f, "two asset combos")
    logger(f, "-------------------------")
    two_asset_combos = common.get_two_asset_combinations(bond_names)
    for combo in two_asset_combos:
        bonds_in_combo = combo.split("-")  # splits bondA-bondB into its components

        # retrieve bond object with calculated prices and stats for first name
        bond1 = bond_calcs[bonds_in_combo[0]]["object"]
        bond2 = bond_calcs[bonds_in_combo[1]]["object"]

        # todo if use_correlation is Matrix then fetch thresholds, make gauss corr mat, do numerical integration
        # fetch the relevant joint transition probability
        joint_probs_lookup = bond1.rating + "|" + bond2.rating  # concatenate bond names to match joint probs file names
        joint_trans_probs = joint_probs_master[joint_probs_lookup]  # lookup joint transition probabilities

        # send bond1, bond2 and joint transition probabilities into function to do looping for price stats
        price_stats = engines.calc_two_asset_portfolio_stats(bond1, bond2, joint_trans_probs)

        bond_calcs[combo] = {"type": "two_asset",
                             "stats": {"pct": price_stats["pct"], "dollar": price_stats["dollar"]},
                             "object": None}

        logger(f, combo)
        # logger(f, "joint trans probs - " + str(joint_trans_probs))
        # logger(f, "price stats pct - " + str(price_stats["pct"]))
        logger(f, "Mean: " + str(round(common.fmt_num(bond_calcs[combo]['stats']['dollar']['mean'], '$', 1, 'MM'), 3)))
        logger(f, "Var: " + str(round(common.fmt_num(bond_calcs[combo]['stats']['dollar']['variance'], '$', 2, 'MM'), 3)))
        logger(f, "-----------------------")

    logger(f, "----------------------")
    logger(f, "List of bond calculations")
    logger(f, str(bond_calcs.keys()))

    # loop through calculations that were done to get portfolio level stuff
    portfolio_mean = 0.0
    portfolio_variance = 0.0
    for calc_name in bond_calcs.keys():

        if bond_calcs[calc_name]["type"] == "single_asset":
            # portfolio mean is sum of the means
            portfolio_mean += bond_calcs[calc_name]["object"].price_stats_dollar["mean"]
            # subtract single asset vars
            # portfolio var is var(p) = var(b1+b2) + var(b2+b3) + var(b1+b3) - var(b1) - var(b2) - var(b3)
            portfolio_variance -= bond_calcs[calc_name]["object"].price_stats_dollar["variance"]

        if bond_calcs[calc_name]["type"] == "two_asset":
            # add the two-asset sub portfolio vars
            portfolio_variance += bond_calcs[calc_name]["stats"]["dollar"]["variance"]

    logger(f, "--------------------")
    logger(f, "Portfolio Results")
    logger(f, "Mean : " + str(portfolio_mean))
    logger(f, "Variance : " + str(portfolio_variance / 1000000**2))
    logger(f, "Std Dev : " + str((portfolio_variance / 1000000**2) * 0.5))

    logger(f, "--------------------")
    logger(f, "Marginal Variances")
    # Calculating marginal variances and standard deviations
    # loop through bonds and adjust mean and variance to exclude bond and sub-portfolios

    # for each single bond in the list
    for bondname in bond_names:

        # marginal variance comes from adjusting portfolio variance
        marginal_var = portfolio_variance

        # find all the single and two asset calculation packages that bond was involved in
        for calc_name in bond_calcs.keys():
            if bondname in calc_name:

                if bond_calcs[calc_name]["type"] == "single_asset":
                    # add the variance back
                    marginal_var += bond_calcs[calc_name]["object"].price_stats_dollar["variance"]

                if bond_calcs[calc_name]["type"] == "two_asset":
                    # subtract the pair-wise variance out
                    marginal_var -= bond_calcs[calc_name]["stats"]["dollar"]["variance"]

        bond_calcs[bondname]["object"].marginal_variance = marginal_var
        logger(f, bondname + ": " + str(marginal_var / 1000000.0**2))


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


def make_random_portfolio(num_securities, portfolio_type, clear_db=False):
    """
    script that will make a random portfolio and insert into DB
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

    # open DB connection
    db_conn = common.open_db_connection()
    cursor = db_conn.cursor()

    # delete existing entrie?
    if clear_db:
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


if __name__ == "__main__":
    example_three_bond_calculation_analytical()
    #make_random_portfolio(10, "Balanced", True)

