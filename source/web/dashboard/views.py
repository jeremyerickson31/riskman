from django.shortcuts import render

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
