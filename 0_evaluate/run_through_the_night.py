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

"""
If ic=0 => absolute sigma => dataPoint variablestd = small error
If ic!=0 rel sigma => common properties => new property value = ln((val/1.5)/(val*1.5))/4
"""


def splitSigmas(df, inputs, observables, must_be_zero, wide=False):
    rel_sigmas = dict()

    for index, row in df.iterrows():
        if row.species in inputs:
            # if row.species not in observables:
            continue
        if row.value in must_be_zero and row.species not in inputs:
            rel_sigmas[row.species] = 5e-14
        elif row.species not in inputs and row.species in observables:
            if wide and row.minconc != row.maxconc:
                rel_sigmas[row.species] = ((row.maxconc-row.minconc)/8)*1e-12
            else:
                rel_sigmas[row.species] = ((row.value*1.5-row.value/2)/8)*1e-12
    return rel_sigmas


def makeBounds(df):
    bounds = dict()
    for index, row in df.iterrows():
        if row.value < 0.1:
            lb = 1e-14
            ub = 1e-13
        else:
            lb = (row.value/2)*1e-12
            ub = (row.value*1.5)*1e-12
        bounds[row.species] = [lb, ub]
    return bounds


def makeBounds2(df):
    bounds = dict()
    for index, row in df.iterrows():
        if row.value < 0.1:
            lb = 1e-14
            ub = 1e-13
        else:
            lb = (row.minconc)*1e-12
            ub = (row.maxconc)*1e-12
        bounds[row.species] = [lb, ub]
    return bounds


def generate_opp_content(xml_folder: str, name: str, num_xmls: list[int], mech_file: str = "7_Krisztian/mech/BCRN6.inp", 
                            yaml_file: str = "7_Krisztian/mech/BCRN6.yaml", time_limit: int = 50, thread_limit: int = 32,
                            settings_tag: str = "systems_biology", solver: str = "cantera", extension: str = ".xml") -> str:
    # Collect all matching XML files for this worksheet
    folder = Path(xml_folder)
    xml_files = sorted(f for f in folder.glob(f"*{name}*{extension}"))

    # Create MECHMOD section
    mechmod = f"""MECHMOD
    USE_NAME         BCRN6
    MECH_FILE        {mech_file}
    COMPILE_cantera  {yaml_file}
    END
    """

    # Create MECHTEST section
    mechtest = f"""MECHTEST
    MECHANISM  BCRN6
    TIME_LIMIT {time_limit}
    THREAD_LIMIT {thread_limit}
    SETTINGS_TAG {settings_tag}
    FALLBACK_TO_DEFAULT_SETTINGS

    SOLVER {solver}
    SAVE_STATES      CSV
    """

    # Add each XML file name
    for xml in num_xmls:
        padded_number = str(xml).zfill(5)
        mechtest += f"      NAME {xml_folder}/stac_{xml}.xml\n"

    mechtest += "END\n"

    return mechmod + "\n" + mechtest


df_species_ics = pd.read_excel('input_files/reactions_ics_finalised.xlsx', sheet_name='icranges')
df_species_ics['value'] = df_species_ics['value'].astype(float)

observables = []
for index, row in df_species_ics.iterrows():
    if row.value > 0:
        observables.append(row.species)
with open('observables.txt', 'w') as f:
    for spec in observables:
        f.write(f"{spec}\n")

# inoputokat ki kell szedni
input_names = ['nS', 'RAP', 'TG', 'dS', 'CCH', 'REF', 'Insulin', 'TG_SERCA', 'mTOR_RAP', 'casp', 'IP3R', 'Baxa', 'tBid']
must_be_zero = ['casp', 'Baxa', 'tBid', 'p53a', 'PUMA']
inputs = dict()
for i in input_names:
    inputs[i] = 0.0
inputs["REF"] = 1.0
inputs["Insulin"] = 1e-10

rel = splitSigmas(df_species_ics, input_names, observables, must_be_zero)
rel_wide = splitSigmas(df_species_ics, input_names, observables, must_be_zero, wide=True)
species = df_species_ics.species.to_list()
only_vars = list(set(species)-set(input_names))
no_inp_species = []

# dataPoints values
columns = list(set(observables)-set(input_names))
columns.sort()
columns.insert(0, 'time')
time = np.linspace(0, 24, 25)

dataPoints = pd.DataFrame(columns=columns)
dataPoints['time'] = time*60

# Fill in the "theoretical" stacionary conentrations
for index, row in df_species_ics.iterrows():
    if row.species in dataPoints.columns:
        if row.value == 0:
            dataPoints.loc[:, row.species] = 1e-13
        else:
            dataPoints.loc[:, row.species] = row.value*1e-12
# dataPoints

df = pd.read_excel('input_files/reactions_ics_finalised.xlsx', sheet_name=None)

bounds = makeBounds(df["Sheet7"])
bounds2 = makeBounds2(df["icranges"])
print(bounds['mTORa'])
print(bounds2['mTORa'])

# Directory to save files
output_directory = '/home/nvme/Opt/7_Krisztian/xml/5000_old'
output_directory2 = '/home/nvme/Opt/7_Krisztian/xml/5000_new'
output_directory3 = '/home/nvme/Opt/7_Krisztian/xml/5000_wide_new'

# Create the directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
if not os.path.exists(output_directory2):
    os.makedirs(output_directory2)
if not os.path.exists(output_directory3):
    os.makedirs(output_directory3)

num_xml = 15000
# dataPoints['PKC'] = dataPoints['PKC'] * 0.1

# np.random.seed(0)

for i in range(5001, num_xml+1):
    file_index = i
    generate_file(file_index, output_directory, only_vars, inputs, bounds, dataPoints, rel)
    generate_file(file_index, output_directory2, only_vars, inputs, bounds2, dataPoints, rel)
    generate_file(file_index, output_directory3, only_vars, inputs, bounds2, dataPoints, rel_wide)
print("job finished")
print(len(only_vars))
print(len(input_names))

xmls = [np.arange(5001, 6001, 1), np.arange(6001, 7001, 1),
        np.arange(7001, 8001, 1), np.arange(8001, 9001, 1),
        np.arange(9001, 10001, 1), np.arange(10001, 11001, 1),
        np.arange(11001, 12001, 1), np.arange(12001, 13001, 1),
        np.arange(13001, 14001, 1), np.arange(14001, 15001, 1)]   # max(num_xmls) <= num_xml kell

date = datetime.datetime.now()
opp_output_dir = "../1_mechtest"
old_opps = []
new_opps = []
new_opps_wide = []

for num in xmls:
    opp_content = generate_opp_content(output_directory, name='stac', num_xmls=num)
    opp_content2 = generate_opp_content(output_directory2, name='stac', num_xmls=num)
    opp_content3 = generate_opp_content(output_directory3, name='stac', num_xmls=num)

    opp_filename = f"2025526_BCRN_corr_{num[-1]}_old.opp"
    opp_filename2 = f"2025526_BCRN_corr_{num[-1]}_new.opp"
    opp_filename3 = f"2025526_BCRN_corr_{num[-1]}_wide_new.opp"

    old_opps.append(opp_filename)
    new_opps.append(opp_filename2)
    new_opps_wide.append(opp_filename3)

    with open(os.path.join(opp_output_dir, opp_filename), "w") as f:
        f.write(opp_content)
    with open(os.path.join(opp_output_dir, opp_filename2), "w") as f:
        f.write(opp_content2)
    with open(os.path.join(opp_output_dir, opp_filename3), "w") as f:
        f.write(opp_content3)

print(f"old ones:\n{old_opps}\nnew ones:\n{new_opps}\nnew wide ones:\n{new_opps_wide}")

parent_path = Path.cwd().parents[1]
print(parent_path)

command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/2025526_BCRN_corr_5000_old.opp"]
print(f"Running: {' '.join(command)}")
with open(f"../logs/2025526/run_log_old5000.txt", "w") as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)

for idx, opp_file in enumerate(old_opps):
    command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{new_opps[idx]}"]
    print(f"Running: {' '.join(command)}")
    with open(f"../logs/2025526/run_log_new{xmls[idx][-1]}.txt", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)

for idx, opp_file in enumerate(old_opps):
    command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{new_opps_wide[idx]}"]
    print(f"Running: {' '.join(command)}")
    with open(f"../logs/2025526/run_log_new_wide{xmls[idx][-1]}.txt", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)

for idx, opp_file in enumerate(old_opps):
    command = ["bin/Release/OptimaPP", f"7_Krisztian/1_mechtest/{opp_file}"]
    print(f"Running: {' '.join(command)}")
    with open(f"../logs/2025526/run_log_old{xmls[idx][-1]}.txt", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parent_path)

