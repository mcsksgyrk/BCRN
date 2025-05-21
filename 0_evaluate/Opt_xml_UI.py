import sys
import os
import numpy as np
import pandas as pd
import jinja2
import bibtexparser
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt
from pathlib import Path
import datetime

# --- Core Classes from Existing Code ---


class Experiment:
    def __init__(self, data_source, # data_source: either a string (path to CSV) or a pandas DataFrame
                 stresses: dict[str, tuple[float, str]],
                 bibtex: str = ""):

        if isinstance(data_source, str):    # Ha .csv filename
            self.name = os.path.splitext(os.path.basename(data_source))[0]
            self.experiment_data = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame): # Ha pandas DataFrame (azaz xlsx worksheet)
            self.name = "worksheet_experiment"
            self.experiment_data = data_source.copy()
        else:
            raise ValueError("data_source must be a file path or a pandas DataFrame")

        self.stresses = stresses
        self.bibtex = self.parse_bibtex(bibtex)
        self.non_species_cols = {"TIME"}
        self.process_data()

    def parse_bibtex(self, bibtex_str):
        parser = bibtexparser.loads(bibtex_str)
        entry = parser.entries[0]  # Assume only one entry is given

        return {
            "author": entry.get("author", ""),
            "title": entry.get("title", ""),
            "journal": entry.get("journal", ""),
            "volume": entry.get("volume", ""),
            "number": entry.get("number", ""),
            "year": entry.get("year", ""),
            "doi": entry.get("doi", entry.get("url", ""))  # fallback if no DOI
        }

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
                quant_Data[s] *= ics[s[0:-3]]
            else:
                quant_Data[s] *= ics[s]
        self.quant_data = quant_Data


class TheoreticalRanges:
    def __init__(self, min_max_path: str, scaling_factor: float, first_species_col: int):
        self.name = os.path.splitext(os.path.basename(min_max_path))[0]
        if min_max_path.endswith('.csv'):
            self.df_ranges = pd.read_csv(min_max_path)
        elif min_max_path.endswith('.xlsx'):
            self.df_ranges = pd.read_excel(min_max_path, sheet_name="testics")
        self.scaling_factor = scaling_factor
        self.df_scaled_ranges = self.df_ranges.select_dtypes(include='number') * self.scaling_factor
        self.bounds = self.get_bounds(first_species_col)

    def get_bounds(self, first_species_col) -> dict[str, tuple[float, float]]:
        bounds = {}
        for col in self.df_scaled_ranges.iloc[:, first_species_col-1:]:
            lb, ub = self.df_scaled_ranges[col][:2]
            if lb == '' or lb < 0:
                lb = 0
            if ub == '' or ub < 0:
                ub = 0
            col = col.replace('x_', '')
            bounds[col.upper()] = [lb, ub]
        return bounds

    def check_compatibility(self, experiment: Experiment) -> None:
        for s in experiment.species:
            if s not in self.bounds.keys():
                self.bounds[s] = [0, 0]


class Simulation:
    def __init__(self, species_range: TheoreticalRanges, experiment: Experiment):
        self.species_range = species_range
        self.experiment = experiment
        self.species_range.check_compatibility(experiment=self.experiment)

    def create_xml_files(self, output_xmls_path: str, num_of_xmls: int, xml_template_path: str) -> None:
        if not os.path.exists(output_xmls_path):
            os.makedirs(output_xmls_path)

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(xml_template_path)))
        self.template = env.get_template(os.path.basename(xml_template_path))

        for i in range(1, num_of_xmls+1):
            self.random_ics = self.get_random_ics()
            self.experiment.quantitated_exp_data(ics=self.random_ics)
            self.make_xml_output(i, output_xmls_path)

    def get_random_ics(self) -> dict[str, float]:
        random_ics = {s: np.random.uniform(*self.species_range.bounds[s]) for s in self.species_range.bounds}
        for s in self.experiment.stresses:
            if self.experiment.stresses[s][1] == "molecular_species":
                random_ics[s] = self.experiment.stresses[s][0]
        random_ics["REF"] = 1.0
        return random_ics

    def make_xml_output(self, file_index: int, output_xmls_path: str) -> None:
        dataPoints = [self.compileDataRow(row.values) for _, row in self.experiment.quant_data.iterrows()]
        output = self.template.render(ics=self.random_ics, variables=self.experiment.species, dataPoints=dataPoints, bib=self.experiment.bibtex)
        filename = f"{self.experiment.bibtex['author'].split()[0][:-1]+'_'+self.experiment.bibtex['year']}_{self.experiment.name}_{file_index:04d}.xml"
        with open(os.path.join(output_xmls_path, filename), 'w') as f:
            f.write(output)

    def compileDataRow(self, dataPoints):
        meas = "".join(f"<{v}>{{:.4e}}</{v}>" for v in self.experiment.experiment_data.columns)
        return f"<dataPoint>{meas.format(*dataPoints)}</dataPoint>"


# --- PyQt UI Class ---

class OptimaSimulatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optima Experiment Simulation Setup")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # === Experiment Section ===
        layout.addWidget(QLabel("Experiment:"))

        self.exp_file_btn = QPushButton("Choose experiment file")
        self.exp_file_btn.clicked.connect(self.choose_exp_file)
        layout.addWidget(self.exp_file_btn)

        self.exp_info_input = QLineEdit()
        self.exp_info_input.setPlaceholderText("Enter stress info: e.g. (molecular_species/starvation RAP 100e-12)")
        layout.addWidget(self.exp_info_input)

        # === Theoretical Range Section ===
        layout.addWidget(QLabel("Theoretical Ranges:"))

        self.range_file_btn = QPushButton("Choose range file")
        self.range_file_btn.clicked.connect(self.choose_range_file)
        layout.addWidget(self.range_file_btn)

        self.scaling_input = QLineEdit()
        self.scaling_input.setPlaceholderText("Enter scaling factor (e.g., 1e-12)")
        layout.addWidget(self.scaling_input)

        self.first_col_input = QSpinBox()
        self.first_col_input.setRange(1, 100)
        self.first_col_input.setPrefix("First species col index: ")
        layout.addWidget(self.first_col_input)

        # === Simulation Section ===
        layout.addWidget(QLabel("Simulation:"))

        self.template_file_btn = QPushButton("Choose template .xml file")
        self.template_file_btn.clicked.connect(self.choose_template_file)
        layout.addWidget(self.template_file_btn)

        self.output_dir_btn = QPushButton("Choose output .xml directory")
        self.output_dir_btn.clicked.connect(self.choose_output_dir)
        layout.addWidget(self.output_dir_btn)

        self.opp_output_dir_btn = QPushButton("Choose output .opp directory")
        self.opp_output_dir_btn.clicked.connect(self.opp_choose_output_dir)
        layout.addWidget(self.opp_output_dir_btn)

        self.num_xml_input = QSpinBox()
        self.num_xml_input.setRange(1, 100)
        self.num_xml_input.setPrefix("# of XMLs: ")
        layout.addWidget(self.num_xml_input)

        # === Run Simulation ===
        self.run_button = QPushButton("Create XML Files")
        self.run_button.clicked.connect(self.create_xmls)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def choose_exp_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Experiment File", "", "Data Files (*.csv *.xlsx)")
        if file:
            self.exp_file_btn.setText(file)

    def choose_range_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Range File", "", "Data Files (*.csv *.xlsx)")
        if file:
            self.range_file_btn.setText(file)

    def choose_template_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Template XML", "", "XML Files (*.xml)")
        if file:
            self.template_file_btn.setText(file)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select .xml Output Directory")
        if directory:
            self.output_dir_btn.setText(directory)
    
    def opp_choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select .opp Output Directory")
        if directory:
            self.opp_output_dir_btn.setText(directory)

# Define a function to generate .opp file content
    def generate_opp_content(self, xml_folder: str, worksheet_name: str, mech_file: str = "7_Krisztian/mech/BCRN6.inp", 
                            yaml_file: str = "7_Krisztian/mech/BCRN6.yaml", time_limit: int = 50, thread_limit: int = 32,
                            settings_tag: str = "systems_biology", solver: str = "cantera", extension: str = ".xml") -> str:
        # Collect all matching XML files for this worksheet
        folder = Path(xml_folder)
        xml_files = sorted(f for f in folder.glob(f"*{worksheet_name}*{extension}"))

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
        for xml in xml_files:
            mechtest += f"      NAME {xml.as_posix()}\n"

        mechtest += "END\n"

        return mechmod + "\n" + mechtest

    def create_xmls(self):
        try:
            exp_xlsx_path = self.exp_file_btn.text()
            range_csv = self.range_file_btn.text()
            xml_template = self.template_file_btn.text()
            output_dir = self.output_dir_btn.text()
            opp_output_dir = self.opp_output_dir_btn.text()
            num_xml = self.num_xml_input.value()
            scaling_factor = float(self.scaling_input.text())
            first_species_col = self.first_col_input.value()

            # Parse stresses
            stress_parts = self.exp_info_input.text().split()
            if len(stress_parts) == 3:
                stresses = {stress_parts[0]: (stress_parts[1], float(stress_parts[2]))}
            elif stress_parts[0] != "molecular_species":
                stresses = {stress_parts[0]: ("", "")}

            # Read all sheets from Excel
            all_sheets = pd.read_excel(exp_xlsx_path, sheet_name=None)  # dict of {sheet_name: DataFrame}

            # Extract BibTeX from the last sheet
            last_sheet_name = list(all_sheets.keys())[-1]
            bibtex_df = all_sheets[last_sheet_name]
            # Ha nem lenne header a BibTex-nel, akk ezzel kell beolvasni a sheetet: bibtex_df = pd.read_excel(exp_xlsx_path, sheet_name=last_sheet_name, header=None)

            # Join all non-empty strings from the first column into a BibTeX string
            bibtex_lines = bibtex_df.iloc[:, 0].dropna().astype(str).tolist()
            bibtex_str = "\n".join(bibtex_lines)

            print("\n", bibtex_str, "\n")

            if len(bibtex_str) == 0:
                QMessageBox.warning(self, "Input Error", "No valid BibTeX found in the last worksheet.")
                return
            
            date = datetime.datetime.now()

            for sheet_name in list(all_sheets.keys())[:-1]:
                df = all_sheets[sheet_name]
                exp = Experiment(df, stresses, bibtex_str)
                exp.name = sheet_name
                rng = TheoreticalRanges(range_csv, scaling_factor, first_species_col)
                sim = Simulation(rng, exp)
                sim.create_xml_files(output_dir, num_xml, xml_template)
                opp_content = self.generate_opp_content(output_dir, sheet_name)  # Create .opp file content
                opp_filename = f"{date.year}{date.month}{date.day}_BCRN_{exp.bibtex['author'].split()[0][:-1]}_{sheet_name}.opp" # Define output .opp file path
                with open(os.path.join(opp_output_dir, opp_filename), "w") as f:
                    f.write(opp_content)

            QMessageBox.information(self, "Success", "XML files have been successfully generated for all worksheets.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OptimaSimulatorUI()
    window.show()
    sys.exit(app.exec_())


# I have to be in the folder where 'bin' is, or rewrite the commands below!!!!!!

# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250222_BCRN_proba.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250123_BCRN_cor.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250223_BCRN_debug.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20240820_BCRN.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/20250224_BCRN_OOPgoverned.opp
# bin/Release/OptimaPP 7_Krisztian/2_sensitivity/20250224_BCRN6.opp
