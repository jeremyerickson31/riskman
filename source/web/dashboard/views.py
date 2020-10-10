from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound

import numpy
import json
import sys
import os

sys.path.append(os.getcwd() + "\\dashboard")
from utils import common
from utils import engines
from utils import config


# Create your views here.

def intro(request):
    context = dict()
    context["nav_tree_id"] = ["nav_tree_id_intro"]
    context["nav_tree_parents"] = ["nav_tree_id_home"]
    return render(request, 'dashboard/intro.html', context)


def equity(request):
    context = dict()
    context["nav_tree_id"] = "nav_tree_id_equity"
    context["nav_tree_parents"] = ["nav_tree_id_riskman"]
    return render(request, 'dashboard/equity.html', context)


def credit_risk(request):
    context = dict()

    # nav tree information
    context["nav_tree_id"] = "nav_tree_id_credit_risk"
    context["nav_tree_parents"] = ["nav_tree_id_riskman", "nav_tree_id_fixedincome"]

    # get list of file names in sample_portfolio_folder to add to dropdown
    portfolios = common.get_available_bond_portfolios()
    context["bond_portfolio_names"] = portfolios
    
    return render(request, 'dashboard/credit_risk.html', context)


def interest_rate_risk(request):
    context = dict()
    context["nav_tree_id"] = "nav_tree_id_interest_rate_risk"
    context["nav_tree_parents"] = ["nav_tree_id_riskman", "nav_tree_id_fixedincome"]
    return render(request, 'dashboard/interest_rate_risk.html', context)


def one_year_transitions(request):
    context = dict()
    context["nav_tree_id"] = "nav_tree_id_one_year_transitions"
    context["nav_tree_parents"] = ["nav_tree_id_tables"]
    return render(request, 'dashboard/one_year_transitions.html', context)


#########################################################
# AJAX endpoints
#########################################################

def ajax_get_trans_matrix(request):
    """
    AJAX function to return the one year transition matrix from the provider
    :return: transition matrix
    """

    response = {'status': 1, 'message': 'OK', 'data': {}}

    if request.method == "POST":
        try:
            matrix_source = request.POST.get("matrix_source")  # gets the value from Matrix Source dropdown
            ordered_keys, matrix_data = common.get_one_year_matrix(matrix_source)
            response["data"]["ordered_keys"] = ordered_keys
            response["data"]["matrix_data"] = matrix_data
        except:
            response["status"] = 0
            response["message"] = "ERROR: couldn't fetch transition matrix for %s" % matrix_source

    else:
        response["status"] = 0
        response["message"] = "ERROR: ajax endpoint {ajax_get_trans_matrix} not configured for GET request"

    if response:
        return HttpResponse(json.dumps(response), content_type="application/json")
    else:
        return HttpResponseNotFound(json.dumps(response), content_type="application/json")


def ajax_get_cred_risk_calcs(request):
    """
    AJAX function to kick off the credit risk analytics calculations
    :return: 
    """

    response = {'status': 1, 'message': 'OK', 'data': {"analytical_graph": dict(),
                                                       "analytical_table": {"columns": list(), "data": list()},
                                                       "analytical_details_table": {"columns": list(), "data": list()},
                                                       "simulation_graph": dict(),
                                                       "simulation_graph_half": dict(),
                                                       "simulation_graph_quarter": dict(),
                                                       "simulation_pctls_table": {"columns": list(), "data": list()}}
                }

    if request.method == "POST":
        try:

            # ##### read in form data #####
            print("reading in form data")
            portfolio_name = request.POST.get("portfolio_name")
            if request.POST.get("calculation_type") == "Analytical + Monte":
                calculation_type = "all"
            else:
                calculation_type = request.POST.get("calculation_type").lower()
            trans_matrix_source = request.POST.get("trans_matrix_source")
            correlation = float(request.POST.get("correlation"))

            # ##### load the sample portfolio submitted from the form #####
            print("reading in bond portfolio")
            bond_list = common.load_bond_portfolio_json(portfolio_name)

            # ##### push bond portfolio and form parameters into engine #####
            print("running credit risk calcs")
            results = engines.run_portfolio_credit_risk(bond_list,
                                                        run_type=calculation_type,
                                                        provider=trans_matrix_source,
                                                        correlation=correlation)

            # ##### ANALYTICAL RESULTS #####
            # ##### make the scatter plot names and colors #####
            ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
            for rating in ratings:
                response["data"]["analytical_graph"][rating] = {"name": rating,
                                                                'color': config.graph_settings['scatter_graph_colors'][
                                                                    rating],
                                                                "data": []}

            # ##### get data from results for the graphs and tables #####
            analytical_col_headers = ["ID", "Name", "Rating", "Mat", "Coup (%)", "Face ($)", "Value ($)", "Mean ($)", "Var ($<sup>2)", "Marginal ($<sup>2)"]
            response["data"]["analytical_table"]["columns"] = [{"title": col} for col in analytical_col_headers]

            analytical_details_col_headers = ["Name", "Rating", "Prices by Rating ($)", "Prices by Rating (%)", "Trans Probs"]
            response["data"]["analytical_details_table"]["columns"] = [{"title": col} for col in analytical_details_col_headers]

            analytical_bond_calcs = results["analytical"]["bond_calcs"]
            bond_id = 0
            for bond_name in analytical_bond_calcs.keys():
                bond = analytical_bond_calcs[bond_name]
                if bond["type"] == "single_asset":
                    bond_id += 1
                    bond = bond["object"]
                    name = bond.name
                    rating = bond.rating
                    maturity = bond.maturity
                    coupon = round(bond.coupon_pct * 100, 2)
                    notional = round(common.fmt_num(bond.notional, "$", 1, "MM"), 5)
                    value = round(common.fmt_num(bond.market_value_dollar, "$", 1, "MM"), 5)
                    mean = round(common.fmt_num(bond.price_stats_dollar["mean"], "$", 1, "MM"), 5)
                    variance = round(common.fmt_num(bond.price_stats_dollar["variance"], "$", 2, "MM"), 5)
                    marg_variance = round(common.fmt_num(bond.marginal_variance, "$", 2, "MM"), 5)
                    pct_std_dev = round(variance**0.5 / value * 100, 3)
                    rating_level_pct_prices_fmt = {
                        key: round(bond.rating_level_prices_pct[key], 3)
                        for key in bond.rating_level_prices_dollar.keys()}
                    rating_level_dollar_prices_fmt = {
                        key: round(common.fmt_num(bond.rating_level_prices_dollar[key], "$", 1, "MM"), 3)
                        for key in bond.rating_level_prices_dollar.keys()}

                    analytical_table_package = [bond_id, name, rating, maturity, coupon, notional, value, mean, variance, marg_variance]
                    analytical_details_table_package = [name, rating,
                                                        str(rating_level_dollar_prices_fmt),
                                                        str(rating_level_pct_prices_fmt),
                                                        str(bond.transition_probs)]
                    graph_package = {"name": bond_name,
                                     "x": value,
                                     "y": pct_std_dev}

                    response["data"]["analytical_table"]["data"].append(analytical_table_package)
                    response["data"]["analytical_details_table"]["data"].append(analytical_details_table_package)
                    response["data"]["analytical_graph"][bond.rating]["data"].append(graph_package)

                else:
                    pass

            response["data"]["analytical_graph"] = [response["data"]["analytical_graph"][rating] for rating in response["data"]["analytical_graph"].keys()]

            # ##### SIMULATION RESULTS #####
            # ##### get simulation results and make into numpy array for adding #####
            sim_results = results["simulation"]["sim_results"]

            portfolio_prices = numpy.array(sim_results["sim_bond_prices"]).sum(axis=0)  # summing down the column adds the bond prices for each sim run
            prices_histogram, bins = numpy.histogram(portfolio_prices, bins=100)
            prices_histogram, bins = list(prices_histogram), list(bins)
            prices_histogram = [float(price) for price in prices_histogram]
            bins = [round(bin / 1000000.0, 3) for bin in bins]
            cumulative_prices = numpy.array(prices_histogram).cumsum()
            cumulative_prices = [int(value) for value in cumulative_prices]

            response["data"]["simulation_graph"]["categories"] = bins
            response["data"]["simulation_graph"]["series"] = prices_histogram
            response["data"]["simulation_graph"]["cumulative_series"] = cumulative_prices
            response["data"]["simulation_graph_half"]["categories"] = bins[0:51]
            response["data"]["simulation_graph_half"]["series"] = prices_histogram[0:51]
            response["data"]["simulation_graph_quarter"]["categories"] = bins[0:26]
            response["data"]["simulation_graph_quarter"]["series"] = prices_histogram[0:26]

            # make percentiles and get percentiles of value distribution"
            portfolio_curr_val = results["analytical"]["portfolio_calcs"]["current_portfolio_value"]
            sim_percentile_columns = ["ID", "Percentile", "Value (MM$)", "% Change"]
            simulation_percentile_table_data = list()
            percentiles = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
            id = 0
            for pctl in percentiles:
                id += 1
                pctl_name = str(pctl*100) + "th %"
                pctl_value = numpy.percentile(portfolio_prices, pctl*100.0)
                pctl_percent_of_currval = round((pctl_value - portfolio_curr_val) / portfolio_curr_val * 100, 3)

                pctl_value = round(pctl_value / 1000000.0, 5)
                simulation_percentile_table_data.append([id, pctl_name, pctl_value, pctl_percent_of_currval])

            response["data"]["simulation_pctls_table"]["data"] = simulation_percentile_table_data
            response["data"]["simulation_pctls_table"]["columns"] = [{"title": col} for col in sim_percentile_columns]

        except:
            response["status"] = 0
            response["message"] = "ERROR: "

    else:
        response["status"] = 0
        response["message"] = "ERROR: ajax endpoint {ajax_cred_risk_analytics} not configured for GET request"

    if response:
        return HttpResponse(json.dumps(response), content_type="application/json")
    else:
        return HttpResponseNotFound(json.dumps(response), content_type="application/json")
