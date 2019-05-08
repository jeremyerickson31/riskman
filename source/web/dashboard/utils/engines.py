# engines.py will hold the main calculation functions

import config
import common
import numpy
import scipy.stats

from datetime import datetime


def calc_rating_transition_thresholds(provider, mu, sigma):
    """
    this is the calculation engine that is used to determine the rating level transition thresholds
    this is dependent only on the one year transition probabiities and the asset return distribution parameters
    it is done recursively by solving for the D migration threshold and using this as an input into other thresholds
    the calculation is explained in the Credit Metrics Technical Document
    :param provider: transition matrix provider to use
    :param mu: asset return distribution mean
    :param sigma: asset return distribution standard deviation
    :return: dictionary of rating level thresholds per rating level (ie: AAA: [AAA, AA, ....])
    """

    engine_name = "calc_rating_transition_thresholds"
    logging = list()  # list for logging engine process. can be printed and stored elsewhere
    rating_level_thresholds = dict()

    logging.append("ENGINE: entered engine: %s" % engine_name)

    logging.append("ENGINE: Check Provider validity")
    supported_providers = common.get_matrix_providers()
    if provider not in supported_providers:
        raise Exception("Parameter Error: Got unsupported Provider: {%s}" % provider)

    logging.append("ENGINE: Check Mu, Sigma Validity")
    if not isinstance(mu, float) or not isinstance(sigma, float):
        raise Exception("Parameter Error: Mu and Sigma must both be of type <float>. Got %s and %s" % (str(type(mu)), str(type(sigma))))

    logging.append("ENGINE: Get one year transition matrix for provider: %s" % provider)
    ordered_rating_levels, matrix = common.get_one_year_matrix(provider)
    rating_indexes = list(ordered_rating_levels.keys())
    rating_indexes.sort()
    rating_levels = [ordered_rating_levels[index] for index in rating_indexes]
    rating_levels.reverse()  # reversed because we start with P(D) and solve going up

    logging.append("ENGINE: making Normal and Inverse Normal Distribution with Mu=%s and Sigma=%s " % (mu, sigma))
    norm = scipy.stats.norm(mu, sigma).cdf
    norm_inv = scipy.stats.norm(mu, sigma).ppf

    logging.append("ENGINE: Begin loop through From-Rating levels")
    for from_rating in rating_levels[1:]:  # excludes transitions FROM D
        rating_level_thresholds[from_rating] = dict()

        logging.append("ENGINE: From-Rating = %s" % from_rating)
        one_year_probs = matrix[from_rating]
        logging.append("ENGINE: One year transitions for %s = %s" % (from_rating, str(one_year_probs)))

        logging.append("ENGINE: Loop through To-Ratings and get thresholds")
        prev_thresh = 0
        for to_rating in rating_levels:

            trans_prob = one_year_probs[to_rating] / 100.0
            if trans_prob == 0.0:
                trans_prob = 0.001  # this is required to avoid an inf or -inf threshold

            logging.append("ENGINE: From-Rating = %s, To-Rating = %s" % (from_rating, to_rating))
            logging.append("ENGINE: Transition Prob from %s to %s = %s" % (from_rating, to_rating, trans_prob))
            logging.append("ENGINE: Previous Threshold = %s" % prev_thresh)

            if to_rating == rating_levels[0]:
                logging.append("ENGINE: Using To-Rating = 'bottom rating' calculation Variant")
                curr_thresh = norm_inv(trans_prob) * sigma + mu
            elif to_rating == rating_levels[-1]:
                logging.append("ENGINE: Using To-Rating = 'top rating' calculation Variant")
                curr_thresh = prev_thresh  # anything above top rating level -1 is upgrade to top rating level
            else:
                logging.append("ENGINE: Using Normal Threshold Calculation")
                curr_thresh = (norm_inv(trans_prob + norm((prev_thresh - mu) / sigma)) * sigma) + mu

            rating_level_thresholds[from_rating][to_rating] = curr_thresh
            logging.append("ENGINE: Calculated Threshold = %s" % curr_thresh)
            logging.append("ENGINE: Next To-Rating")
            prev_thresh = curr_thresh

        logging.append("ENGINE: Completed All To-Ratings")
        logging.append("ENGINE: Completed Threshold Calculations for %s" % from_rating)
        logging.append("ENGINE: Next From-Rating")

    logging.append("ENGINE: Completed All From-Ratings")
    logging.append("ENGINE: All Threshold Calculations Complete")

    return rating_level_thresholds, logging


def threshold_numerical_integration(thresholds_1, thresholds_2, gauss_corr_mat):
    """
    this function calculates the joint rating transition probabilities by way of numerical integration of a bivariate
    standard normal distribution
    required input is a dictionary of rating transition thresholds, one for each credit exposure and a correlation
    assumption for the bivariate normal distribution
    :param thresholds_1: thresholds for first credit exposure ie: 'CCC': {'D': -0.85, 'CCC': 1.02, ...}
    :param thresholds_2: thresholds for second credit exposure ie: 'CCC': {'D': -0.85, 'CCC': 1.02, ...}
    :param gauss_corr_mat: 2x2 correlation amtrix for bivariate normal distribution, must be 2x2 numpy array
    :return: joint transition probabilities matrix for a pair of ratings
    """

    engine_name = "threshold_numerical_integration"
    logging = list()  # list for logging engine process. can be printed and stored elsewhere
    joint_trans_probs = dict()

    logging.append("ENGINE: entered engine: %s" % engine_name)

    # check that the matrices have the same keys. otherwise not square matrix and thats weird
    if not list(thresholds_1.keys()) == list(thresholds_2.keys()):
        raise Exception("Parameter Error: thresholds are not the same shape. Should have the same keys")

    # check that the corr_mat is a 2x2 numpy array
    if type(gauss_corr_mat) == numpy.ndarray:
        if gauss_corr_mat.shape == (2, 2):
            pass
        else:
            raise Exception("Parameter Dimension Error: gauss_corr_mat needs to be (2,2) numpy array. "
                            "Got %s" % str(gauss_corr_mat.shape))
    else:
        raise Exception("Parameter Type Error: gauss_corr_mat needs to be numpy array. "
                        "Got %s" % str(type(gauss_corr_mat)))

    # get the ordered keys from the one year matrix file and sort
    ordered_keys = common.get_ordered_rating_keys()
    key_ints = list(ordered_keys.keys())
    key_ints = [int(key) for key in key_ints]  # make the str numbers to integers to guarantee correct sorting
    key_ints.sort()
    key_ints.reverse()  # will start with D and go up

    # hey_top and key_bottom used to know upper and lower bounds of integration
    key_bottom = key_ints[0]
    key_top = key_ints[-1]

    # these limits set the outer most limits for the integration, instead of -INF and INF
    upper_limit_max = 10.0
    lower_limit_max = -10.0

    # get mu 1x1 matrix for bivariate gauss
    mu = common.make_bivariate_gauss_mu_mat()

    logging.append("ENGINE: Begin numerical integration loop")
    for key_1 in key_ints:

        if key_1 == key_bottom:
            bond1_lower_limit = lower_limit_max
            bond1_upper_limit = thresholds_1[ordered_keys[str(key_1)]]
            bond1_lower_rating = "-INF"
            bond1_upper_rating = ordered_keys[str(key_1)]
        elif key_1 == key_top:
            bond1_lower_limit = thresholds_1[ordered_keys[str(key_1)]]
            bond1_upper_limit = upper_limit_max
            bond1_lower_rating = ordered_keys[str(key_1)]
            bond1_upper_rating = "+INF"
        else:
            bond1_lower_limit = thresholds_1[ordered_keys[str(key_1 + 1)]]
            bond1_upper_limit = thresholds_1[ordered_keys[str(key_1)]]
            bond1_lower_rating = ordered_keys[str(key_1 + 1)]
            bond1_upper_rating = ordered_keys[str(key_1)]

        print("ENGINE: Bond 1 integration limits {%s: %s, %s: %s}"
                       % (bond1_lower_rating, bond1_lower_limit, bond1_upper_rating, bond1_upper_limit))

        # the upper rating label of the integration is the TO-rating
        # put results in dictionary
        if bond1_upper_rating == "+INF":
            bond1_to_rating = bond1_lower_rating
        else:
            bond1_to_rating = bond1_upper_rating

        joint_trans_probs[bond1_to_rating] = dict()

        for key_2 in key_ints:

            if key_2 == key_bottom:
                bond2_lower_limit = lower_limit_max
                bond2_upper_limit = thresholds_2[ordered_keys[str(key_2)]]
                bond2_lower_rating = "-INF"
                bond2_upper_rating = ordered_keys[str(key_2)]
            elif key_2 == key_top:
                bond2_lower_limit = thresholds_2[ordered_keys[str(key_2)]]
                bond2_upper_limit = upper_limit_max
                bond2_lower_rating = ordered_keys[str(key_2)]
                bond2_upper_rating = "+INF"
            else:
                bond2_lower_limit = thresholds_2[ordered_keys[str(key_2 + 1)]]
                bond2_upper_limit = thresholds_2[ordered_keys[str(key_2)]]
                bond2_lower_rating = ordered_keys[str(key_2 + 1)]
                bond2_upper_rating = ordered_keys[str(key_2)]

            print("ENGINE: Bond 2 integration limits {%s: %s, %s: %s}"
                           % (bond2_lower_rating, bond2_lower_limit, bond2_upper_rating, bond2_upper_limit))

            # set variables for input in mvn
            lower_bound = numpy.array([bond1_lower_limit, bond2_lower_limit])
            upper_bound = numpy.array([bond1_upper_limit, bond2_upper_limit])

            # numerical 2D integration for bivariate gauss
            p, i = scipy.stats.mvn.mvnun(lower_bound, upper_bound, mu, gauss_corr_mat)
            print("ENGINE: Integration results = " + str(p))

            # put results in dictionary
            if bond2_upper_rating == "+INF":
                bond2_to_rating = bond2_lower_rating
            else:
                bond2_to_rating = bond2_upper_rating

            # add joint transition probability to output dictionary
            joint_trans_probs[bond1_to_rating][bond2_to_rating] = p

    return joint_trans_probs
