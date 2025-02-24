import numpy as np
import pandas as pd
import jinja2
import os


class Experiment:
    def __init__(self, experiment_csv_path: str,
                 stresses: dict[str, tuple[float, str]],
                 year: str = "", author: str = "", doi: str = ""):
        
        self.name = os.path.splitext(os.path.basename(experiment_csv_path))[0]
        self.stresses = stresses
        self.year = year
        self.author = author
        self.doi = doi
        self.experiment_data = pd.read_csv(experiment_csv_path)
        self.non_species_cols = {"TIME"}
        self.process_data()

    def __str__(self):
        return f"Experiment\n File used for object init.: {self.name}.csv\nStresses: {self.stresses}\nYear: {self.year}\nAuthor: {self.author}\nDOI: {self.doi}\n\n"

    def process_data(self) -> None:
        self.experiment_data.columns = [col.upper() for col in self.experiment_data.columns]
        self.experiment_data.rename(columns={'TIME': 'time'}, inplace=True)
        self.experiment_data = self.experiment_data.dropna()
        self.species = [v for v in self.experiment_data.columns if v.upper() not in self.non_species_cols and "STD" not in v.upper()]

    def quantitated_exp_data(self, ics: dict[str, float]) -> None:
        quant_Data = self.experiment_data.copy()
        species_and_std = [col for col in quant_Data.columns if col.upper() not in self.non_species_cols]
        for s in species_and_std:
            if 'STD' in s.upper():
                quant_Data[s] *= ics[s[0:-3]]  # Scaling standard devs
            else:
                quant_Data[s] *= ics[s]    # Scaling relative measurement values
        self.quant_data = quant_Data


class TheoreticalRanges:
    def __init__(self, min_max_csv_path: str, scaling_factor: float, first_species_col: int):
        self.name = os.path.splitext(os.path.basename(min_max_csv_path))[0]
        self.df_ranges = pd.read_csv(min_max_csv_path)
        self.scaling_factor = scaling_factor
        self.df_scaled_ranges = self.df_ranges.select_dtypes(include='number') * self.scaling_factor
        self.bounds = self.get_bounds(first_species_col)

    def __str__(self):
        first_ten_bounds = dict(list(self.bounds.items())[:10])
        return f"TheoreticalRanges\n File used for object init.: {self.name}.csv\nScaling Factor: {self.scaling_factor}\nFirst 10 cases in protein bounds: {first_ten_bounds}\n\n"

    # first_species_column is the index of the first column with protein data in the min_max_csv file.
    # Indexing of columns for first_species_col starts from 1!!!
    def get_bounds(self, first_species_col) -> dict[str, tuple[float, float]]:
        bounds = {}
        for col in self.df_scaled_ranges.iloc[:, first_species_col-1:]:
            lb, ub = self.df_scaled_ranges[col][:2]
            if lb == '' or lb < 0:
                lb = 0
                print(f"Warning: {col} has a negative lower bound in the provided .csv for ranges. Setting lb to 0.\n")
            if ub == '' or ub < 0:
                ub = 0
                print(f"Warning: {col} has a negative upper bound in the provided .csv for ranges. Setting ub to 0.\n")
            col = col.replace('x_', '')  # remove 'x_' prefix from column names to match protein names in the experiment data
            bounds[col.upper()] = [lb, ub]
        return bounds

    def check_compatibility(self, experiment: Experiment) -> None:
        bool_compatible = True
        for s in experiment.species:
            if s not in self.bounds.keys():
                self.bounds[s] = [0, 0]
                print(f"Warning: No specified ranges for {s} are found in the provided .csv for cc. ranges. Setting bounds to 0.\n")
                bool_compatible = False
        if bool_compatible:
            print(f"Theoretical ranges are compatible with the experiment data. All experiment species are found in {self.name}.csv\n")


class Simulation:
    def __init__(self, species_range: TheoreticalRanges = "", experiment: Experiment = ""):
        self.species_range = species_range
        self.experiment = experiment
        self.species_range.check_compatibility(experiment=self.experiment)

    def __str__(self):
        return f"Simulation\nSpecies Range used: {self.species_range.name}.csv\nExperiment data used: {self.experiment.name}.csv\n\n"

    def create_xml_files(self, output_xmls_path: str, num_of_xmls: int) -> None:
        if not os.path.exists(output_xmls_path):
            os.makedirs(output_xmls_path)
            print(f"Output XMLs directory had to be created. The provided {output_xmls_path} was not found.\n")

        for i in range(1, num_of_xmls+1):
            file_index = i
            self.random_ics = self.get_random_ics()
            self.experiment.quantitated_exp_data(ics=self.random_ics)       # Creates the quantitated (i.e., not just fold changes) data as a field in the experiment object
            self.make_xml_output(file_index, output_xmls_path)

    def get_random_ics(self) -> dict[str, float]:
        
        random_ics = dict()
        
        for s in self.species_range.bounds:
            lb, ub = self.species_range.bounds[s]
            random_ics[s] = np.random.uniform(lb, ub)

        for s in self.experiment.stresses:
            if self.experiment.stresses[s][1] == "molecular_species":    # i.e., a real molecule, not like a "starvation level", or sth like that
                random_ics[s] = self.experiment.stresses[s][0]
        
        random_ics["REF"] = 1.0
        return random_ics

    def make_xml_output(self, file_index: int, output_xmls_path: str) -> None:
        dataPoints = []
        for i, row in self.experiment.quant_data.iterrows():
            dataPoints.append(self.compileDataRow(row.values))
        output = self.generateOutput(dataPoints)
        filename = self.generateFileName(file_index)
        with open(os.path.join(output_xmls_path, filename), 'w') as f:
            f.write(output)

    def compileDataRow(self, dataPoints):
        meas = ""
        for v in self.experiment.experiment_data.columns:
            meas = meas+"<%s>" % v + "{:.4e}" + "</%s>" % v
        start = "<dataPoint>"
        close = "</dataPoint>"
        row = start+meas.format(*dataPoints)+close
        return row
    
    def generateOutput(self, dataPoints):
        file_loader = jinja2.FileSystemLoader('.')
        env = jinja2.Environment(loader=file_loader)
        template = env.get_template('7_Krisztian/0_evaluate/input_files/data_w_std.xml')
        # megszorozza a számolt hibával, a maximum értékét a mérésnek
        output = template.render(ics=self.random_ics, variables=self.experiment.species,
                                dataPoints=dataPoints)
        return output

    def generateFileName(self, file_index: int) -> str:
        stresses = "".join(f"{k}_{str(v[0])}" for k, v in self.experiment.stresses.items())
        return f"{self.experiment.author+self.experiment.year}_{self.experiment.name}_{self.experiment.name}_{stresses}_{file_index:04d}.xml"


Holczer_rap = Experiment("7_Krisztian/0_evaluate/input_files/rap.csv", {"RAP": (100e-12, "molecular_species")}, "2019", "Holczer")

ranges = TheoreticalRanges("7_Krisztian/0_evaluate/input_files/min_max_ranges.csv", 1e-12, 5)

sim1 = Simulation(ranges, Holczer_rap)

sim1.create_xml_files('/home/szupernikusz/TDK/Opt/7_Krisztian/xml/OOPgovernor_Holczer2019/', 20)

print(Holczer_rap)
print(ranges)



# You have to be in the folder where 'bin' is!!!!!!

# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250222_BCRN_proba.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250123_BCRN_cor.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250223_BCRN_debug.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20240820_BCRN.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250224_BCRN_OOPgoverned.opp

