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
                # todo curr_thresh calc can yield NaN due to norm_inv exploding (only in moodys)
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
    :param gauss_corr_mat: 2x2 correlation matrix for bivariate normal distribution, must be 2x2 numpy array
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

    # get matrix sum tolerance
    tol = common.get_matrix_sum_tolerance()
    # the sum of all matrix entries should ~ 1
    matrix_sum = 0.0

    logging.append("ENGINE: Begin numerical integration loop")
    # get integration limits for bond1
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

        logging.append("ENGINE: Bond 1 integration limits {%s: %s, %s: %s}"
                       % (bond1_lower_rating, bond1_lower_limit, bond1_upper_rating, bond1_upper_limit))

        # the upper rating label of the integration is the TO-rating
        # put results in dictionary
        if bond1_upper_rating == "+INF":
            bond1_to_rating = bond1_lower_rating
        else:
            bond1_to_rating = bond1_upper_rating
        joint_trans_probs[bond1_to_rating] = dict()

        # get integration limits for bond2
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

            # put results in dictionary
            if bond2_upper_rating == "+INF":
                bond2_to_rating = bond2_lower_rating
            else:
                bond2_to_rating = bond2_upper_rating

            logging.append("ENGINE: Bond 2 integration limits {%s: %s, %s: %s}"
                           % (bond2_lower_rating, bond2_lower_limit, bond2_upper_rating, bond2_upper_limit))

            # set variables for input in mvn
            lower_bound = numpy.array([bond1_lower_limit, bond2_lower_limit])
            upper_bound = numpy.array([bond1_upper_limit, bond2_upper_limit])

            # numerical 2D integration for bivariate gauss
            # todo fix that p can be NaN (only for moodys and SP)
            p, i = scipy.stats.mvn.mvnun(lower_bound, upper_bound, mu, gauss_corr_mat)
            logging.append("ENGINE: Integration results = " + str(p))

            # add joint transition probability to output dictionary
            joint_trans_probs[bond1_to_rating][bond2_to_rating] = p

            # add p to matrix_sum
            matrix_sum += p

    logging.append("ENGINE: Sum of Matrix Probabilities = %s" % matrix_sum)

    if 1.0 - matrix_sum <= tol:
        logging.append("ENGINE: Sum of Matrix Probabilities within tolerance? --> PASS")
    else:
        logging.append("ENGINE: Sum of Matrix Probabilities within tolerance? --> ***FAIL***")

    return joint_trans_probs, logging


def forward_interest_rate_repricing(bond, forward_curve):
    """
    This function calculates the price of the bond under the forward rates provided
    :param bond: a Bond class object with associated attributes like par, coupon, maturity etc
    :param forward_curve: interest rates to use in the discounted cash flow calculation
    :return: populates the Bond.value_under_forwards
    """
    # todo turn this into a generic discounted cash flow function

    engine_name = "forward_interest_rate_repricing"
    logging = list()

    if not isinstance(bond, Bond):
        raise Exception("Type Error: bond parameter must be a Bond class object with associated attributes")

    if bond.maturity == 1:
        price = bond.par + bond.coupon_dollar
    else:
        # discount cash flows
        price = 0.0
        for i, r in enumerate(forward_curve):

            # if on last rate then receive coupon plus principal
            if i == len(forward_curve) - 1:
                price += (bond.coupon_dollar + bond.par) / (1 + r) ** (i + 1)
            else:
                price += bond.coupon_dollar / (1 + r) ** (i + 1)

        # from today we calc price at end of year 1. Receive end of year 1 coupon + PV of future coupons and par
        price += bond.coupon_dollar

    return price, logging


def rand_to_rating(initial_rating, provider, rand_num):
    """
    this function accepts a random number between 0 and 1 (or a list of randoms), applies the normal inverse function,
    and finds what rating that corresponds to for a given provider transition thresholds
    :param initial_rating: initial rating
    :param rand_num: float between 0 and 1, or list of floats
    :param provider: Credit Metrics, Moodys, SP
    :return: single rating as string or list of ratings
    """

    if not isinstance(rand_num, float) and not isinstance(rand_num, list):
        raise Exception("Parameter Error: rand_num must be a float. Got " + str(type(rand_num)))

    # load transition thresholds
    trans_thresholds = common.load_transition_thresholds(provider)
    initial_rating_thresholds = trans_thresholds[initial_rating]

    # get ordered ratings so we can go from D --> AAA
    ratings = common.get_ordered_rating_keys()
    rating_keys = list(ratings.keys())
    rating_keys.sort()
    rating_keys.reverse()  # results in [8,7,6,...] one nunber for each major rating level in oneyeartransitions.json)

    # convert random number between 0 and 1 to a standard normal value
    if isinstance(rand_num, float):
        transition_values = [rand_num]
    if isinstance(rand_num, list):
        transition_values = rand_num

    results = []
    for trans_value in transition_values:

        for key in rating_keys:
            rating_level = ratings[key]
            threshold = initial_rating_thresholds[rating_level]

            if trans_value < threshold:
                final_rating = rating_level
                break
            else:
                continue
        else:
            # if the less than is never hit then the rand number is really high and the highest rating is the result
            final_rating = ratings[rating_keys[-1]]
        results.append(final_rating)

    if isinstance(rand_num, float):
        results = results[0]

    return results


def calc_two_asset_portfolio_stats(bond1, bond2, joint_trans_probs):
    """
    This function takes two Bond class instances along with a joint probability matrix and will calculate the
    value and mean and variance of the pairwise portfolio
    :param bond1: a Bond Class instance
    :param bond2: a Bond Class instance
    :param joint_trans_probs: distionary of joint rating transition probabilities
    :return: dictionary of stats {"stats_pct":{}, "stats_dollar": {} }
    """

    if not isinstance(bond1, Bond) or not isinstance(bond2, Bond):
        raise Exception("Type Error: Parameters bond1 and bond2 must be of type Bond with associated properties")

    if set(joint_trans_probs.keys()) != set(bond1.rating_level_prices_pct.keys()):
        raise Exception("Key Mismatch Error: bond1, bond2 and joint_trans_prob must have same set of dictionary keys")

    mean_pct = 0.0
    mean_dollar = 0.0
    for rating_1 in bond1.rating_level_prices_pct.keys():
        for rating_2 in bond2.rating_level_prices_pct.keys():
            bond1_price_pct = bond1.rating_level_prices_pct[rating_1]
            bond1_price_dollar = bond1.rating_level_prices_dollar[rating_1]
            bond2_price_pct = bond2.rating_level_prices_pct[rating_2]
            bond2_price_dollar = bond2.rating_level_prices_dollar[rating_2]

            joint_trans_prob = round(joint_trans_probs[rating_1][rating_2], 5)

            mean_pct += joint_trans_prob * (bond1_price_pct + bond2_price_pct)
            mean_dollar += joint_trans_prob * (bond1_price_dollar + bond2_price_dollar)

    variance_pct = 0.0
    variance_dollar = 0.0
    for rating_1 in bond1.rating_level_prices_pct.keys():
        for rating_2 in bond2.rating_level_prices_pct.keys():
            bond1_price_pct = bond1.rating_level_prices_pct[rating_1]
            bond1_price_dollar = bond1.rating_level_prices_dollar[rating_1]
            bond2_price_pct = bond2.rating_level_prices_pct[rating_2]
            bond2_price_dollar = bond2.rating_level_prices_dollar[rating_2]

            joint_trans_prob = round(joint_trans_probs[rating_1][rating_2], 5)

            variance_pct += joint_trans_prob * ((bond1_price_pct + bond2_price_pct) - mean_pct)**2
            variance_dollar += joint_trans_prob * ((bond1_price_dollar + bond2_price_dollar) - mean_dollar)**2

    stats = {"pct": {}, "dollar": {}}

    stats["pct"]["mean"] = mean_pct
    stats["pct"]["variance"] = variance_pct
    stats["pct"]["std_dev"] = variance_pct ** 0.5

    stats["dollar"]["mean"] = mean_dollar
    stats["dollar"]["variance"] = variance_dollar
    stats["dollar"]["std_dev"] = variance_dollar ** 0.5

    return stats


def run_portfolio_credit_risk(bonds, run_type="all", provider="Credit Metrics", correlation=0.3):
    """
    this is the main engine that will rin the portfolio credit risk analytics

    :param bonds: a list of dictionaries with bond properties
    :param run_type: Can be 'analytical' or 'simulation'
    :param provider: one of the transition matrix providers
    :param correlation: either float in approved list or a correlation matrix
    :return: output differs for each run_type
    """

    engine_name = "run_portfolio_credit_risk"
    logging = list()  # list for logging engine process. can be printed and stored elsewhere

    logging.append("ENGINE: entered engine: %s" % engine_name)

    # ######################## parameter validation #########################
    # validate bonds
    logging.append("ENGINE: Validating List of Bonds and doing Bond Properties Check")
    if not isinstance(bonds, list):
        raise Exception("Parameter Error: bond must be a list of dictionaries")
    missing_properties = common.run_bond_properties_check(bonds)
    if missing_properties:
        properties_error = "Bond Properties Error: List of bonds has missing properties needed for calcs\n"
        properties_error += str(missing_properties)
        raise Exception(properties_error)

    # validate run_type
    logging.append("ENGINE: Validating Run Type")
    if run_type not in ["analytical", "simulation", "all"]:
        raise Exception("Parameter Error: run_type must be 'analytical', 'simulation', 'all'")

    # validate provider
    logging.append("ENGINE: Validating Transition Matrix Provider")
    provider_list = common.get_matrix_providers()
    if provider not in provider_list:
        raise Exception("Parameter Error: provider must be one of the following %s" % str(provider_list))

    # validate correlation
    # could be a single float or a numpy square matrix, for now only works as a single float
    # todo extend for numpy square matrix
    logging.append("ENGINE: Validating Correlation")
    if not isinstance(correlation, float):
        raise Exception("Parameter Error: Correlation must be a single float")
    # ######################## parameter validation #########################

    # ###################### bond re-pricing ######################
    # do this now because rating level prices are used in both Analytical and Simulation
    # rating level interest rate curves for bond repricing under rating level scenarios
    logging.append("ENGINE: Fetching rating level interest rates for bond repricing")
    forward_rates = common.get_interest_rate_curves()
    bond_objects = []
    for bond_properties in bonds:
        bond = Bond(bond_properties)  # initialize the bond object
        bond.get_transition_probabilities(provider)  # fetch transition probs for given provider and self.rating
        bond.calc_prices_under_forwards(forward_rates)  # use provided forward rates to do re-pricing
        bond_objects.append(bond)

    # run calculations as per input
    credit_risk_results = dict()

    if run_type == "analytical" or run_type == "all":
        bond_calcs, portfolio_calcs, analytic_logs = run_credit_risk_analytical(bond_objects, provider, correlation)
        credit_risk_results["analytical"] = {"bond_calcs": bond_calcs,
                                             "portfolio_calcs": portfolio_calcs,
                                             "logs": analytic_logs}

    if run_type == "simulation" or run_type == "all":
        sim_results, simulation_logs = run_credit_risk_simulation(bond_objects, provider, correlation)
        credit_risk_results["simulation"] = {"sim_results": sim_results}

    return credit_risk_results


def run_credit_risk_analytical(bond_list, provider, correlation):
    """
    this is the engine that will run the portfolio credit risk under the analytical approach
    The calculation is the Analytical approach which involves the following:
        (1) for each bond, calculate the value in each possible state
        (2) apply probabilities for each state to get a mean and variance for each bond
        (3) identify each pair of bonds and calculate pair-wise sub-portfolio value in each possible state
        (4) apply joint probabilities to get sub-portfolio mean and variance
        (5) combine individual mean values to get portfolio mean value
        (6) combine pair-wise variances with individual variances to get portfolio variance

    :param bond_list: a list of Bond objects
    :param provider: one of the transition matrix providers
    :param correlation: either float in approved list or a correlation matrix
    :return: results of the analytical calcs: bond_calcs (dict), portfolio_calcs (dict), logging (list)
    """
    #
    # ########################  begin calculations  #########################
    #

    logging = list()

    # get master file of joint probabilities for provider/correlation pair
    logging.append("ENGINE: Fetching pre-made joint probability matrix for %s and %s" % (provider, correlation))
    joint_probs_master = common.get_provider_correlation_joint_probs(provider, correlation)

    logging.append("ENGINE: Performing Single Bond Calcs")
    bond_names = [bond.name for bond in bond_list]  # list of names to use for making two asset sub portfolios
    bond_calcs = dict()  # dictionary for holding the single asset and two asset calculations that have been done
    for bond in bond_list:
        bond.calc_price_stats()  # apply transition probabilities to prices to get mean and variance

        # add this bond object to the dictionary of bond objects that we have done calculations for
        bond_calcs[bond.name] = {"type": "single_asset", "stats": None, "object": bond}

    logging.append("ENGINE: Performing Two Asset Sub-Portfolio Calcs")
    two_asset_combos = common.get_two_asset_combinations(bond_names)
    for combo in two_asset_combos:
        bonds_in_combo = combo.split("-")  # splits bondA-bondB into its components

        # retrieve bond object with calculated prices and stats for first name
        bond1 = bond_calcs[bonds_in_combo[0]]["object"]
        bond2 = bond_calcs[bonds_in_combo[1]]["object"]

        # fetch the relevant joint transition probability
        joint_probs_lookup = bond1.rating + "|" + bond2.rating  # concatenate bond names to match joint probs file names
        joint_trans_probs = joint_probs_master[joint_probs_lookup]  # lookup joint transition probabilities

        # send bond1, bond2 and joint transition probabilities into function to do looping for price stats
        price_stats = calc_two_asset_portfolio_stats(bond1, bond2, joint_trans_probs)

        # add two-asset portfolio stats to the dictionary of bond calcs
        bond_calcs[combo] = {"type": "two_asset",
                             "stats": {"pct": price_stats["pct"], "dollar": price_stats["dollar"]},
                             "object": None}

    logging.append("ENGINE: Performing Portfolio Calculations")
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

    portfolio_calcs = {"mean": portfolio_mean, "variance": portfolio_variance}

    # loop through to get the marginal variance for each single asset in bond calcs
    for bond_name in bond_names:
        portfolio_variance_without = portfolio_variance
        for calc_name in bond_calcs.keys():
            if bond_name in calc_name:
                if bond_calcs[calc_name]["type"] == "single_asset":
                    # individual variances will be added back
                    portfolio_variance_without += bond_calcs[calc_name]["object"].price_stats_dollar["variance"]

                if bond_calcs[calc_name]["type"] == "two_asset":
                    # two asset variances will be subtracted back out
                    portfolio_variance_without -= bond_calcs[calc_name]["stats"]["dollar"]["variance"]
            else:
                pass

        bond_calcs[bond_name]["object"].marginal_variance = portfolio_variance - portfolio_variance_without

    return bond_calcs, portfolio_calcs, logging


def run_credit_risk_simulation(bond_list, provider, correlation):
    """
    this is the engine that will run the portfolio credit risk under the simulation approach

    :param bond_list: a list of bond objects
    :param provider: one of the transition matrix providers
    :param correlation: either float in approved list or a correlation matrix
    :return: results of the simulation calcs: TBD
    """
    logging = list()
    sim_runs = 10000
    sim_results = {"sim_bond_ratings": None, "sim_bond_prices": None}

    logging.append("ENGINE: Performing Simulation Calculation")

    logging.append("ENGINE: Making Matrix of Correlated Random Numbers")
    correlated_randoms = common.make_flat_square_correlated_random_matrix(correlation, len(bond_list), sim_runs).tolist()

    logging.append("ENGINE: Beginning Transition Threshold and Price Lookup for Each Random")
    all_bond_ratings = list()
    all_bond_prices = list()
    for bond, rand_list in zip(bond_list, correlated_randoms):
        logging.append("ENGINE: Performing transition and price lookups for " + bond.name)
        simulation_bond_ratings = rand_to_rating(bond.rating, provider, rand_list)
        simulation_bond_prices = [bond.rating_level_prices_dollar[rating] for rating in simulation_bond_ratings]

        all_bond_ratings.append(simulation_bond_ratings)
        all_bond_prices.append(simulation_bond_prices)

    sim_results["sim_bond_ratings"] = all_bond_ratings
    sim_results["sim_bond_prices"] = all_bond_prices

    return sim_results, logging


class Bond:
    """
    This is the Bond Class
    It is an easy way to keep all information and calculations about the bond tied to the Bond object
    The methods of this bond don't return variables but instead set certain Bond attributes
    """

    def __init__(self, properties):
        """
        initialize the Bond object attributes
        """
        # variable for holding logging information on what this bond object is doing
        self.class_name = "Class {Bond}: "
        self.logs = list()

        self.log_action("Setting Bond Class Attributes")
        # main attributes of the bond
        self.name = properties["bond_name"]
        self.par = properties["par"]
        self.coupon_pct = properties["coupon"]
        self.coupon_dollar = self.par * self.coupon_pct
        self.notional = properties["notional"]
        self.maturity = properties["maturity"]
        self.rating = properties["rating"]
        self.seniority = properties["seniority"]
        self.market_value_dollar = None
        self.market_value_pct = None

        self.log_action("Attributes Set {par, coupon_pct, coupon_dollar, maturity, rating, seniority")

        # attribute placeholder for new values in forward rate scenarios
        self.transition_probs = None  # will be transition probabilities pulled for a certain provider
        self.rating_level_prices_pct = dict()  # will have {"AAA": price, ... "D": price"}
        self.rating_level_prices_dollar = dict()  # is the pct variable times notional
        self.price_stats_pct = dict()  # mean , variance, std dev based on price
        self.price_stats_dollar = dict()  # mean, variance, std dev based on price * notional
        self.marginal_variance = None
        self.marginal_standard_dev = None

    def log_action(self, text):
        """
        Function to add some text to the list of actions in the self.logs attribute
        :param text: some text to. either string or list of strings
        """
        timestamp = str(datetime.now())
        if isinstance(text, str):
            self.logs.append(timestamp + " " + self.class_name + text)
        if isinstance(text, list):
            for entry in text:
                self.logs.append(timestamp + " " + self.class_name + str(entry))

    def calc_prices_under_forwards(self, forwards):
        """
        function to apply each forward interest rate curve
        :param forwards: dictionary of forward interest rate curves on a rating level basis
        """
        self.log_action("Calculating Bond price under rating level forward interest rate scenarios")

        for rating_level in forwards.keys():

            # convert the rate number from float to percent
            forward_rates = [rate / 100.00 for rate in forwards[rating_level]]
            # a bond with Maturity remaining has maturity -1 cash flows to discount + 1 coupon at end of year 1
            forward_rate_segment = forward_rates[:self.maturity - 1]

            self.log_action("Re-pricing for rating level " + rating_level)
            price, logs = forward_interest_rate_repricing(self, forward_rate_segment)
            self.rating_level_prices_pct[rating_level] = price  # per 100 of par
            self.rating_level_prices_dollar[rating_level] = (price / 100.00) * self.notional
            self.logs += logs

        # getting the price in the default event
        # call function to apply recoveries_in_default
        # todo will need to incorporate a stochastic factor for recovery in default when doing the simulation
        self.log_action("Re-pricing for rating level D")
        recovery_stats = common.get_recovery_in_default(self.seniority)
        self.rating_level_prices_pct["D"] = recovery_stats["mean"]
        self.rating_level_prices_dollar["D"] = (recovery_stats["mean"] / 100.00) * self.notional

        # set current market value equal to the price under the rating scenario where the rating doesn't change
        self.market_value_dollar = self.rating_level_prices_dollar[self.rating]
        self.market_value_pct = self.rating_level_prices_pct[self.rating]

    def get_transition_probabilities(self, provider):
        """
        function to get the transition probabilities from the providers matrix for the bond's rating
        :param provider: Credit Metrics, Moodys etc
        """
        probabilities = common.get_transition_probs(provider, self.rating)
        self.transition_probs = probabilities

    def calc_price_stats(self):
        """
        This function takes the rating level bond prices and calculates the mean, variance and std dev
        This is done on a price basis and a dollar basis (price * notional)
        """

        if not self.rating_level_prices_pct.keys() == self.transition_probs.keys():
            raise Exception("Key Mismatch Error: rating_level_prices and transition_probs have non-matching keys\n"
                            "Keys must match to perform price stats calculations\n"
                            "Rating Level Prices Keys = %s\n"
                            "Transition Prob Keys = %s" % (self.rating_level_prices_pct.keys(), self.transition_probs.keys()))

        """
        # this block shows how to make a dictionary to a list then to an array then broadcast to matrix
        # this is neat but not used
        ordered_keys = common.get_ordered_rating_keys()
        keys = [int(key) for key in ordered_keys]
        keys.sort()
        ordered_ratings = [ordered_keys[str(num)] for num in keys]
        price_pct_array = numpy.array([self.rating_level_prices_pct[rating] for rating in ordered_ratings])
        price_pct_matrix = price_pct_array.reshape(len(price_pct_array), 1).repeat(len(price_pct_array), axis=1)
        print(price_pct_matrix + price_pct_matrix)
        """

        self.log_action("Calculate mean and std dev of rating level prices")
        mean_pct = 0.0
        for rating_level in self.rating_level_prices_pct.keys():
            price_pct = self.rating_level_prices_pct[rating_level] / 100.00  # prices pct is like 104.63 per 100 par
            prob = self.transition_probs[rating_level] / 100.00  # probability is like 93.4%
            mean_pct += prob * price_pct
        mean_dollar = mean_pct * self.notional

        variance_pct = 0.0
        for rating_level in self.rating_level_prices_pct.keys():
            price_pct = self.rating_level_prices_pct[rating_level] / 100.00  # prices pct is like 104.63 per par
            prob = self.transition_probs[rating_level] / 100.00  # probability is like 93.4%
            variance_pct += prob * ((price_pct - mean_pct) ** 2)
        variance_dollar = variance_pct * self.notional**2

        self.price_stats_pct["mean"] = mean_pct
        self.price_stats_pct["variance"] = variance_pct
        self.price_stats_pct["std_dev"] = variance_pct ** 0.5

        self.price_stats_dollar["mean"] = mean_dollar
        self.price_stats_dollar["variance"] = variance_dollar
        self.price_stats_dollar["std_dev"] = variance_dollar ** 0.5



