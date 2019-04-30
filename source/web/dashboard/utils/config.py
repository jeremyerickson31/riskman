
import os

current_directory = os.getcwd()

############################################
# certain config settings for the model runs
############################################
model_inputs = dict()

# locations of important files
model_inputs["one_year_matrix_file"] = current_directory + "\\dashboard\\utils\\params\\oneyeartransitions.json"
model_inputs["transition_thresholds"] = current_directory + "\\dashboard\\utils\\params\\transition_thresholds.json"
model_inputs["pre_calc_joint_matrics"] = current_directory + "\\dashboard\\utils\\params\\pre_calc_joint_matrices.json"
model_inputs["joint_transitions"] = current_directory + "\\dashboard\\utils\\params\\"

# default assumptions for the asset return normal distribution
model_inputs["asset_return_distribution"] = {"mu": 0, "sigma": 1}

# default correlation assumptions
model_inputs["correlations"] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

############################################
# certain config settings for other stuff
############################################

settings = dict()

# location of folder to dump script logs
settings["script_log_loc"] = current_directory + "\\script_logs"
