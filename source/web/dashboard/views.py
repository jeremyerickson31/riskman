from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound

import json
import sys
import os

sys.path.append(os.getcwd() + "\\dashboard")
from utils import common


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
    context["nav_tree_id"] = "nav_tree_id_credit_risk"
    context["nav_tree_parents"] = ["nav_tree_id_riskman", "nav_tree_id_fixedincome"]
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
        matrix_source = request.POST.get("matrix_source")  # gets the value from Matrix Source dropdown
        ordered_keys, matrix_data = common.get_one_year_matrix(matrix_source)
        response["data"]["ordered_keys"] = ordered_keys
        response["data"]["matrix_data"] = matrix_data
    else:
        response["status"] = 0
        response["message"] = "ERROR: failed to fetch transition matrix"

    if response:
        return HttpResponse(json.dumps(response), content_type="application/json")
    else:
        return HttpResponseNotFound(json.dumps(response), content_type="application/json")
