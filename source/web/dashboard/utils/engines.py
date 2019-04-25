# engines.py will hold the main calculation functions

import config
import common
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
    logging = list()  # if log file is specified, list will be sent to common.script_logger otherwise just print
    rating_level_thresholds = dict()

    logging.append("ENGINE: entered engine: %s" % engine_name)
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

        logging.append("ENGINE: Completed Threshold Calculations for %s" % from_rating)
        logging.append("ENGINE: Next From-Rating")

    logging.append("ENGINE: All Threshold Calculations Complete")

    return rating_level_thresholds, logging
