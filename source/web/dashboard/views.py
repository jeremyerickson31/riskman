from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound

import json
import sys
import os

sys.path.append(os.getcwd() + "\\dashboard")
from utils import common
from utils import engines


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

    response = {'status': 1, 'message': 'OK', 'data': {"analytical_graph": list(),
                                                       "analytical_table": {"columns": list(), "data": list()},
                                                       "simulation_graph": dict()}}

    if request.method == "POST":
        try:

            # read in form data
            print("reading in form data")
            portfolio_name = request.POST.get("portfolio_name")
            if request.POST.get("calculation_type") == "Analytical + Monte":
                calculation_type = "all"
            else:
                calculation_type = request.POST.get("calculation_type").lower()
            trans_matrix_source = request.POST.get("trans_matrix_source")
            correlation = float(request.POST.get("correlation"))

            # load the sample portfolio submitted from the form
            print("reading in bond portfolio")
            bond_list = common.load_bond_portfolio_json(portfolio_name)

            # push bond portfolio and form parameters into engine
            print("running credit risk calcs")
            results = engines.run_portfolio_credit_risk(bond_list,
                                                        run_type=calculation_type,
                                                        provider=trans_matrix_source,
                                                        correlation=correlation)
            print(results)

            # get data for the analytical table results
            col_headers = ["Name", "Rating", "Mat", "Coup (%)", "Face ($)", "Value ($)", "Mean ($)", "Var ($<sup>2)", "Marginal ($<sup>2)"]
            response["data"]["analytical_table"]["columns"] = [{"title": col} for col in col_headers]

            analytical_bond_calcs = results["analytical"]["bond_calcs"]
            for bond_name in analytical_bond_calcs.keys():
                bond = analytical_bond_calcs[bond_name]
                if bond["type"] == "single_asset":
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

                    table_package = [name, rating, maturity, coupon, notional, value, mean, variance, marg_variance]
                    graph_package = {"name": bond_name,
                                     "x": value,
                                     "y": pct_std_dev}

                    response["data"]["analytical_table"]["data"].append(table_package)
                    response["data"]["analytical_graph"].append(graph_package)
                else:
                    pass
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
