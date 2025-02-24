
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
        self.non_species_cols = {"TIME"}                            # Excluding columns that are not protein or prot<std> measurements
        self.process_data()

    def __str__(self):
        return f"Experiment\n File used for object init.: {self.name}.csv\nStresses: {self.stresses}\nYear: {self.year}\nAuthor: {self.author}\nDOI: {self.doi}\n\n"

    def process_data(self) -> None:
        self.experiment_data.columns = [col.upper() for col in self.experiment_data.columns]
        self.experiment_data.rename(columns={'TIME': 'time'}, inplace=True)                                 # .xml works with OPTIMA only if time is lowercase
        self.experiment_data = self.experiment_data.dropna()
        self.species = [v for v in self.experiment_data.columns if v.upper() not in self.non_species_cols and "STD" not in v.upper()]           # variable declarations for the datagroup part of the xml

    def quantitated_exp_data(self, ics: dict[str, float]) -> None:
        quant_Data = self.experiment_data.copy()
        species_and_std = [col for col in quant_Data.columns if col.upper() not in self.non_species_cols]
        for s in species_and_std:
            if 'STD' in s.upper():
                quant_Data[s] *= ics[s[0:-3]]  # Scaling relative standard devs with random initial conditions
            else:
                quant_Data[s] *= ics[s]    # Scaling relative measurement values with random initial conditions
        self.quant_data = quant_Data       # --> if fold change was measured --> first row of prot cc. columns should = 1 * ics[s]


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
        self.species_range.check_compatibility(experiment=self.experiment)  # Check if all the experiment species are found in the theoretical ranges .csv file  

    def __str__(self):
        return f"Simulation\nSpecies Range used: {self.species_range.name}.csv\nExperiment data used: {self.experiment.name}.csv\n\n"

    def create_xml_files(self, output_xmls_path: str, num_of_xmls: int, xml_template_path: str) -> None:
        
        if not os.path.exists(output_xmls_path):
            os.makedirs(output_xmls_path)
            print(f"Output XMLs directory had to be created. The provided {output_xmls_path} was not found.\n")
        
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(xml_template_path)))
        self.template = env.get_template(os.path.basename(xml_template_path))
        for i in range(1, num_of_xmls+1):
            file_index = i
            self.random_ics = self.get_random_ics()
            self.experiment.quantitated_exp_data(ics=self.random_ics)       # Creates the quantitated (i.e., not just fold changes) data as a field in the experiment object for this instance of Simulation
            self.make_xml_output(file_index, output_xmls_path)

    def get_random_ics(self) -> dict[str, float]:
        
        random_ics = dict()
        
        for s in self.species_range.bounds:
            lb, ub = self.species_range.bounds[s]
            random_ics[s] = np.random.uniform(lb, ub)

        for s in self.experiment.stresses:
            if self.experiment.stresses[s][1] == "molecular_species":    # i.e., a real molecule, not like a "starvation level", or sth like that
                random_ics[s] = self.experiment.stresses[s][0]           # if it's a molecule, take the IC from the experiment data ("bemeresi koncentracio")
                                                                         # This will put it into the xml file as a part of species
        random_ics["REF"] = 1.0                                          # This should always be 1 for OPTIMA to work, right?
        return random_ics

    def make_xml_output(self, file_index: int, output_xmls_path: str) -> None:
        dataPoints = []
        for i, row in self.experiment.quant_data.iterrows():             # This is the quantitated (i.e., scaled with ics) experiment data  
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
        output = self.template.render(ics=self.random_ics, variables=self.experiment.species,
                                dataPoints=dataPoints)
        return output

    def generateFileName(self, file_index: int) -> str:             # I think this could be improved
        stresses = "".join(f"{k}_{str(v[0])}" for k, v in self.experiment.stresses.items())
        return f"{self.experiment.author+self.experiment.year}_{self.experiment.name}_{self.experiment.name}_{stresses}_{file_index:04d}.xml"

    def run_simulation(self, dot_opp_file: str) -> None:
        # This function should run the simulation using the provided dot_opp_file
        # Maybe a new class for running
        # self.run = OptimaSimRunner.run_simulation(<initializing variables>)
        pass


class OptimaSimRunner:
    def opp_exists(self, dot_opp_file: str = "") -> None:
        # This function should check if an OPTIMA (.opp) file exists at the provided path
        # or create it if it doesn't
        if dot_opp_file == "":
            self.create_opp_file(path, dot_opp_file)
            self.run_simulation(dot_opp_file)
        else:
            self.run_simulation(dot_opp_file)
        pass
    
    def create_opp_file(self, path: str, dot_opp_file: str) -> None:
        # This function should create an OPTIMA (.opp) file using the provided dot_opp_file
        # and handle any potential errors or exceptions that may occur
        pass

    def run_simulation(self, dot_opp_file: str) -> None:
        # This function should run the simulation using the provided dot_opp_file
        # and call opp_exists() to check if the dot_opp_file exists, and create it if it doesn't exist
        # then run the simulation
        pass


def main():
    print("\n=== WELCOME TO THE EXPERIMENT SIMULATION SETUP ===\n")
    
    # === GET EXPERIMENT DATA ===
    while True:
        experiment_csv_path = input("Enter the path to the experiment CSV file: ").strip()
        if os.path.exists(experiment_csv_path):
            break
        print("Invalid path! Please enter a valid path to an existing CSV file.")

    stresses = {}
    print("""
Define stresses for the experiment. Enter each stress in the format: NAME VALUE STRESS_TYPE
STRESS_TYPE can be either 'molecular_species' (for stress caused by a molecule with a defined cc., e.g., RAP),
or 'arbitrary_stress' (for stress that's more vague, e.g., aa. starvation)
""")

    print("For example: RAP 100e-12 molecular_species")
    print("Type 'done' when finished.\n")

    while True:
        stress_input = input("Enter stress (or type 'done' to finish): ").strip()
        if stress_input.lower() == "done":
            break
        parts = stress_input.split()
        if len(parts) != 3:
            print("Invalid format! Please enter in the format: NAME VALUE UNIT")
            continue
        name, value, unit = parts
        try:
            value = float(value)
            stresses[name] = (value, unit)
        except ValueError:
            print("Invalid number! Please enter a valid numerical value for the stress.")

    year = input("Enter the experiment year: ").strip()
    author = input("Enter the author's name: ").strip()
    doi = input("Enter the DOI (or press enter to skip): ").strip()

    experiment = Experiment(experiment_csv_path, stresses, year, author, doi)
    print("\nExperiment object created successfully!\n")

    # === GET THEORETICAL RANGES DATA ===
    while True:
        min_max_csv_path = input("Enter the path to the theoretical ranges CSV file: ").strip()
        if os.path.exists(min_max_csv_path):
            break
        print("Invalid path! Please enter a valid path to an existing CSV file.")

    while True:
        try:
            scaling_factor = float(input("Enter the scaling factor (e.g., 1e-12): ").strip())
            break
        except ValueError:
            print("Invalid number! Please enter a valid numerical value.")

    while True:
        try:
            first_species_col = int(input("Enter the column index of the first species (starting from 1): ").strip())
            if first_species_col > 0:
                break
            print("Column index must be a positive integer!")
        except ValueError:
            print("Invalid number! Please enter a valid integer.")

    ranges = TheoreticalRanges(min_max_csv_path, scaling_factor, first_species_col)
    print("\nTheoreticalRanges object created successfully!\n")

    simulation = Simulation(ranges, experiment)
    print("\nSimulation object created successfully!\n")

    # === GENERATE XML FILES ===
    while True:
        output_xmls_path = input("Enter the directory path for the output XML files: ").strip()
        if not os.path.exists(output_xmls_path):
            os.makedirs(output_xmls_path)
            print("Directory created successfully!")
        break

    while True:
        try:
            num_of_xmls = int(input("Enter the number of XML files to generate: ").strip())
            if num_of_xmls > 0:
                break
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    while True:
        xml_template_path = input("Enter the path for the template XML file: ").strip()
        if not os.path.exists(output_xmls_path):
            print(f"File is not found at {xml_template_path}. Please provide a valid path.")
        break

    print("\nGenerating XML files... This may take a moment.")
    simulation.create_xml_files(output_xmls_path, num_of_xmls, xml_template_path)
    print(f"\nXML files have been successfully generated in {output_xmls_path}\n")

    # === FINAL SUMMARY ===
    print("\n=== SUMMARY ===")
    print(experiment)
    print(ranges)
    print(simulation)


# main() is executed when the script is run.
# Comment this out if you want to run from code.
# Important: the script was written so that it accepts input .csv files in the format of min_max_ranges.csv (for TheoreticalRanges), and rap.csv (for Experiment).
if __name__ == "__main__":
    main()


# To run from code, uncomment the following lines:

# Holczer_rap = Experiment("path_to rap.csv", {"RAP": (100e-12, "molecular_species")}, "2019", "Holczer")
# ranges = TheoreticalRanges("path to min_max_ranges.csv", 1e-12, 5)
# sim1 = Simulation(ranges, Holczer_rap)
# sim1.create_xml_files('xml output path', 20)
# print(Holczer_rap)
# print(ranges)
# print(sim1)


# I have to be in the folder where 'bin' is, or rewrite the commands below!!!!!!

# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250222_BCRN_proba.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250123_BCRN_cor.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250223_BCRN_debug.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20240820_BCRN.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250224_BCRN_OOPgoverned.opp
# bin/Release/OptimaPP 7_Krisztian/2_sensitivity/20250224_BCRN6.opp
