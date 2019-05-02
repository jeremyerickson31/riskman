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
        for to_rating in rating_levels[:-1]:  # includes transitions TO D but doesn't consider AAA threshold
            # any asset return over the AA threshold is an upgrade to AAA

            trans_prob = one_year_probs[to_rating] / 100.0
            if trans_prob == 0.0:
                trans_prob = 0.001  # this is required to avoid an inf or -inf threshold

            logging.append("ENGINE: From-Rating = %s, To-Rating = %s" % (from_rating, to_rating))
            logging.append("ENGINE: Transition Prob from %s to %s = %s" % (from_rating, to_rating, trans_prob))
            logging.append("ENGINE: Previous Threshold = %s" % prev_thresh)

            if to_rating == "D":
                logging.append("ENGINE: Using To-Rating = D calculation Variant")
                curr_thresh = norm_inv(trans_prob) * sigma + mu
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


def threshold_numerical_integration(thresholds_1, thresholds_2, correlation):
    """
    this function calculates the joint rating transition probabilities by way of numerical integration of a bivariate
    standard normal distribution
    required input is a dictionary of rating transition thresholds, one for each credit exposure and a correlation
    assumption for the bivariate normal distribution
    :param thresholds_1: thresholds for first credit exposure ie: 'CCC': {'D': -0.85, 'CCC': 1.02, ...}
    :param thresholds_2: thresholds for second credit exposure ie: 'CCC': {'D': -0.85, 'CCC': 1.02, ...}
    :param correlation: correlation parameter used in the bivariate normal distribution
    :return: joint transition probabilities matrix for a pair of ratings
    """

    engine_name = "threshold_numerical_integration"
    logging = list()  # list for logging engine process. can be printed and stored elsewhere
    joint_trans_probs = dict()

    """
        
        lower_bound = numpy.array([-10, -10])
    
        upper_bound = numpy.array([.1, -.2])
    
        mu = numpy.array([-.3, .17])
    
        corr_mat = numpy.array([[1.2,.35],[.35,2.1]])
        
        p, i = mvn.mvnun(lower_bound, upper_bound, mu, corr_mat)
        print(p)
    """
    return None
