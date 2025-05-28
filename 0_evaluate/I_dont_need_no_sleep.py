import pandas as pd
import numpy as np
import datetime
from filegenerators import generate_file
pd.options.display.float_format = '{:.2e}'.format
from pathlib import Path
from CovCor_calc import OptimaMechtest, OptimaOutput, OptimaSensitivity
import seaborn as sns
import matplotlib.pyplot as plt
import os
import subprocess

xmls = [np.arange(5001, 6001, 1), np.arange(6001, 7001, 1),
        np.arange(7001, 8001, 1), np.arange(8001, 9001, 1),
        np.arange(9001, 10001, 1), np.arange(10001, 11001, 1),
        np.arange(11001, 12001, 1), np.arange(12001, 13001, 1),
        np.arange(13001, 14001, 1), np.arange(14001, 15001, 1)]   # max(num_xmls) <= num_xml kell

opp_output_dir = "../1_mechtest"
old_opps = []
new_opps = []
new_opps_wide = []

for num in xmls:
    opp_filename = f"2025526_BCRN_corr_{num[-1]}_old.opp"
    opp_filename2 = f"2025526_BCRN_corr_{num[-1]}_new.opp"
    opp_filename3 = f"2025526_BCRN_corr_{num[-1]}_wide_new.opp"

    old_opps.append(opp_filename)
    new_opps.append(opp_filename2)
    new_opps_wide.append(opp_filename3)

print(f"old ones:\n{old_opps}\nnew ones:\n{new_opps}\nnew wide ones:\n{new_opps_wide}")

parent_path = Path.cwd().parents[1]
print(parent_path)

command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{old_opps[-2]}"]
print(f"Running: {' '.join(command)}")
with open(f"../logs/2025526/run_log_old{xmls[-2][-1]}.txt", "w") as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)

command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{old_opps[-1]}"]
print(f"Running: {' '.join(command)}")
with open(f"../logs/2025526/run_log_old{xmls[-1][-1]}.txt", "w") as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)


"""for idx, opp_file in enumerate(old_opps):
    command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{new_opps_wide[idx]}"]
    print(f"Running: {' '.join(command)}")
    with open(f"../logs/2025526/run_log_new_wide{xmls[idx][-1]}.txt", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)

for idx, opp_file in enumerate(old_opps):
    command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{opp_file}"]
    print(f"Running: {' '.join(command)}")
    with open(f"../logs/2025526/run_log_old{xmls[idx][-1]}.txt", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)"""