
import os

current_directory = os.getcwd()

############################################
# certain config settings for the model runs
############################################
model_inputs = dict()

# locations of important files for calculations
model_inputs["one_year_matrix_file"] = current_directory + "\\dashboard\\utils\\params\\oneyeartransitions.json"
model_inputs["transition_thresholds_folder"] = current_directory + "\\dashboard\\utils\\params\\transition_thresholds\\"
model_inputs["joint_matrices_folder"] = current_directory + "\\dashboard\\utils\\params\\joint_trans_prob_matrices\\"
mdoel_inputs["recoveries_in_default"] = current_directory + "\\dashboard\\utils\\params\\recoveries_in_default.json"

# default assumptions for the asset return normal distribution
model_inputs["asset_return_distribution"] = {"mu": 0.0, "sigma": 1.0}

# default correlation assumptions
model_inputs["correlations"] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# tolerance level for transition matrix sum = 1
model_inputs["matrix_sum_tolerance"] = 0.000001

############################################
# certain config settings for other stuff
############################################

settings = dict()

# location of folder to dump script logs
settings["script_log_loc"] = current_directory + "\\script_logs"
