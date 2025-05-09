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

# --- Core Classes from Existing Code ---


class Experiment:
    def __init__(self, experiment_csv_path: str,
                 stresses: dict[str, tuple[float, str]],
                 bibtex: str = ""):

        self.name = os.path.splitext(os.path.basename(experiment_csv_path))[0]
        self.stresses = stresses
        self.bibtex = self.parse_bibtex(bibtex)
        self.experiment_data = pd.read_csv(experiment_csv_path)
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
    def __init__(self, min_max_csv_path: str, scaling_factor: float, first_species_col: int):
        self.name = os.path.splitext(os.path.basename(min_max_csv_path))[0]
        self.df_ranges = pd.read_csv(min_max_csv_path)
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

        self.exp_file_btn = QPushButton("Choose .csv file")
        self.exp_file_btn.clicked.connect(self.choose_exp_file)
        layout.addWidget(self.exp_file_btn)

        self.exp_info_input = QLineEdit()
        self.exp_info_input.setPlaceholderText("Enter stress info: e.g. RAP 100e-12 molecular_species")
        layout.addWidget(self.exp_info_input)

        self.bibtex_input = QTextEdit()
        self.bibtex_input.setPlaceholderText("Enter BibTex")
        layout.addWidget(self.bibtex_input)

        # === Theoretical Range Section ===
        layout.addWidget(QLabel("Theoretical Ranges:"))

        self.range_file_btn = QPushButton("Choose .csv file")
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

        self.num_xml_input = QSpinBox()
        self.num_xml_input.setRange(1, 100)
        self.num_xml_input.setPrefix("# of XMLs: ")
        layout.addWidget(self.num_xml_input)

        # === Run Simulation ===
        self.run_button = QPushButton("Create XML Files")
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def choose_exp_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Experiment CSV", "", "CSV Files (*.csv)")
        if file:
            self.exp_file_btn.setText(file)

    def choose_range_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Range CSV", "", "CSV Files (*.csv)")
        if file:
            self.range_file_btn.setText(file)

    def choose_template_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Template XML", "", "XML Files (*.xml)")
        if file:
            self.template_file_btn.setText(file)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_btn.setText(directory)

    def run_simulation(self):
        try:
            exp_csv = self.exp_file_btn.text()
            range_csv = self.range_file_btn.text()
            xml_template = self.template_file_btn.text()
            output_dir = self.output_dir_btn.text()
            num_xml = self.num_xml_input.value()
            scaling_factor = float(self.scaling_input.text())
            first_species_col = self.first_col_input.value()

            # Parse stresses
            stress_parts = self.exp_info_input.text().split()
            stresses = {stress_parts[0]: (float(stress_parts[1]), stress_parts[2])} if len(stress_parts) == 3 else {}

            bibtex_parts = self.bibtex_input.toPlainText()
            if len(bibtex_parts) < 1:
                QMessageBox.warning(self, "Input Error", "Please enter a valid BibTex.")
                return

            exp = Experiment(exp_csv, stresses, bibtex_parts)
            rng = TheoreticalRanges(range_csv, scaling_factor, first_species_col)
            sim = Simulation(rng, exp)

            sim.create_xml_files(output_dir, num_xml, xml_template)
            QMessageBox.information(self, "Success", "XML files have been successfully generated.")
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
