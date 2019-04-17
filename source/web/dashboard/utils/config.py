
import os

current_directory = os.getcwd()

model_inputs = dict()

# locations of important files
model_inputs["one_year_matrix_file"] = current_directory + "\\dashboard\\utils\\params\\oneyeartransitions.json"

# default assumptions for the asset return normal distribution
model_inputs["asset_return"] = {"mu": 0, "sigma": 1}

# default correlation assumptions
model_inputs["correlations"] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
