from django.conf.urls import url
from . import views

urlpatterns = [url(r'intro', views.intro, name='intro'),
               url(r'equity', views.equity, name='equity'),
               url(r'interest_rate_risk', views.interest_rate_risk, name='interest_rate_risk'),
               url(r'credit_risk', views.credit_risk, name='credit_risk'),
               url(r'one_year_transitions', views.one_year_transitions, name='one_year_transitions'),
               url(r'ajax_get_trans_matrix', views.ajax_get_trans_matrix, name='ajax_get_trans_matrix'),
               url(r'ajax_cred_risk_analytics', views.ajax_cred_risk_analytics, name='ajax_cred_risk_analytics')
               ]
