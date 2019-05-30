*********** PROJECT Description ***********

This project aims to implement, in a scaled down fashion, the Credit Metrics approach
to modelling fixed income credit risk. The full technical document can be found online.
Additionally, extensions may include fixed income interest rate risk management procedures,
as well as certain equity risk management procedures

*********** PROJECT NAVIGATION ***********
source/web
    - holds the relevant code for the dashboard and supporting scripts and engines

source/web/run_script.py
    - serves as the main access point to various scripts including:
        - script to build transition thresholds and joint transition probabilites
        - script to test two-bond analytical approach
        - script to test the two-bond Monte Carlo approach

source/web/dashboard
    - views.py and urls.py are the urls and server endpoints for the Django powered dashboard

source/web/dashboard/static
    - holds the HTML, CSS and HighCharts that power the dashboard

source/web/dashboard/utils
    - params folder: holds static parameters used in the credit risk calculations
    - common.py : gnereic helper functions
    - config.py : certain configuration settings like params file locations, parameter options etc
    - engines.py: holds the main calculations used by the script and dashboard
