import sys
import os
import numpy as np
import pandas as pd
import jinja2
import bibtexparser
import glob
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QSpinBox, QTabWidget,
    QComboBox)
from PyQt5.QtCore import Qt
from pathlib import Path
import datetime

# --- Core Classes from Existing Code ---


class TheoreticalRanges:
    def __init__(self, min_max_path: str, scaling_factor: float):
        self.name = os.path.splitext(os.path.basename(min_max_path))[0]
        if min_max_path.endswith('.csv'):
            self.df_ranges = pd.read_csv(min_max_path)
        elif min_max_path.endswith('.xlsx'):
            self.df_ranges = pd.read_excel(min_max_path, sheet_name="icranges")
        self.scaling_factor = scaling_factor
        self.df_scaled_ranges = self.df_ranges.select_dtypes(include='number') * self.scaling_factor
        self.bounds = self.get_bounds()

    def __str__(self):
        return f"The bounds are: \n{self.bounds}\n"

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {}
        for row in self.df_ranges.iterrows():
            lb = row[1]["minconc"] * self.scaling_factor
            ub = row[1]["maxconc"] * self.scaling_factor
            if lb == '' or lb < 0:
                lb = 0
            if ub == '' or ub < 0:
                ub = 0
            bounds[str(row[1][0]).upper()] = [lb, ub]   # row[1][0] is the species name
        return bounds


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
        existing_cols = [c.upper() for c in quant_Data.columns]

        # Add STD column for each species that does not have an STD column
        for col in quant_Data.columns[1:]:
            if not col.upper().endswith("STD") and str(col).upper()+"STD" not in existing_cols:
                elso = (max(quant_Data[col]) - min(quant_Data[col]))/8
                masodik = np.mean(quant_Data[col])/8
                #print(f"\nAz elso: {elso}, \n A masodik: {masodik}\n")
                std = max(elso, masodik)  # the formula from discord
                idx = quant_Data.columns.get_loc(col)   # find the position of the species column in the current DataFrame
                quant_Data.insert(loc=idx+1, column=col+'STD', value=std) # insert the new STD column right *after* it

        species_and_std = [col for col in quant_Data.columns if col.upper() not in self.non_species_cols]

        for s in species_and_std: 
            if 'STD' in s.upper():
                quant_Data[s] *= ics[s[0:-3]]
            else:
                quant_Data[s] *= ics[s]
        self.quant_data = quant_Data

    def check_compatibility(self, ranges: TheoreticalRanges) -> None:
        omitted = [s for s in self.species if s not in ranges.bounds]
        if omitted:
            # drop them
            self.species = [s for s in self.species if s in ranges.bounds]
            self.experiment_data.drop(columns=omitted, inplace=True)
        return omitted


class Simulation:
    def __init__(self, species_range: TheoreticalRanges, experiment: Experiment):
        self.species_range = species_range
        self.experiment = experiment
        self.omitted = self.experiment.check_compatibility(ranges=self.species_range)

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
        #print(f"Random initial concentrations: {random_ics}\n")
        return random_ics

    def make_xml_output(self, file_index: int, output_xmls_path: str) -> None:
        dataPoints = [self.compileDataRow(row.values) for _, row in self.experiment.quant_data.iterrows()]
        output = self.template.render(ics=self.random_ics, variables=self.experiment.species, dataPoints=dataPoints, bib=self.experiment.bibtex)
        filename = f"{self.experiment.bibtex['author'].split()[0][:-1]+'_'+self.experiment.bibtex['year']}_{self.experiment.name}_{file_index:04d}.xml"
        with open(os.path.join(output_xmls_path, filename), 'w') as f:
            f.write(output)

    def compileDataRow(self, dataPoints):
        meas = "".join(f"<{v}>{{:.4e}}</{v}>" for v in self.experiment.quant_data)
        return f"<dataPoint>{meas.format(*dataPoints)}</dataPoint>"


class RangeSimWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        # === Theoretical Range Section ===
        layout.addWidget(QLabel("Theoretical Ranges:"))
        self.range_file_btn = QPushButton("Choose range file")
        self.range_file_btn.clicked.connect(self.choose_range_file)
        layout.addWidget(self.range_file_btn)

        self.scaling_input = QLineEdit()
        self.scaling_input.setText("1e-12")
        self.scaling_input.setPlaceholderText("Enter scaling factor (e.g., 1e-12)")
        layout.addWidget(self.scaling_input)

        # === Simulation Section ===
        layout.addWidget(QLabel("Input mechanism:"))
        self.inp_file_btn = QPushButton("Choose input file")
        self.yaml_file_btn = QPushButton("Choose template .yaml file")
        layout.addWidget(self.inp_file_btn)
        layout.addWidget(self.yaml_file_btn)

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # find the root of the root directory of the GUI (i.e., 7_Krisztian)
        mech_dir = self.find_mech_dir(root_dir)

        if mech_dir is not None:    # Set an init. .inp and .yaml file if they exist
            inp_path = self.init_inp_file(mech_dir)
            yaml_path = self.init_yaml_file(mech_dir)
            self.inp_file_btn.setText(inp_path)
            self.yaml_file_btn.setText(yaml_path)

        self.yaml_file_btn.clicked.connect(self.choose_yaml_file)
        self.inp_file_btn.clicked.connect(self.choose_input_file)
        
        layout.addWidget(QLabel("Generate:"))
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
        self.num_xml_input.setRange(1, 10000)
        self.num_xml_input.valueChanged.connect(self.on_num_xmls_changed)
        self.num_xml_input.setPrefix("# of XMLs: ")
        layout.addWidget(self.num_xml_input)

        self.setLayout(layout)

    def on_num_xmls_changed(self, val):
        print(f"{self.objectName()} wants {val} XMLs")
        # store in self.num_xml or emit a signal…

    def choose_range_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Range File", "", "Data Files (*.csv *.xlsx)")
        if file:
            self.range_file_btn.setText(file)
            btn = self.sender()              # <-- the button that was clicked
            btn.setText(file)

    def choose_input_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select input file", "", "Data Files (*.inp)")
        if file:
            self.inp_file_btn.setText(file)
            btn = self.sender()
            btn.setText(file)

    def choose_yaml_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select yaml file", "", "Data Files (*.yaml)")
        if file:
            self.yaml_file_btn.setText(file)
            btn = self.sender()
            btn.setText(file)

    def choose_template_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Template XML", "", "XML Files (*.xml)")
        if file:
            self.template_file_btn.setText(file)
            btn = self.sender()
            btn.setText(file)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select .xml Output Directory")
        if directory:
            self.output_dir_btn.setText(directory)
            btn = self.sender()
            btn.setText(directory)

    def opp_choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select .opp Output Directory")
        if directory:
            self.opp_output_dir_btn.setText(directory)
            btn = self.sender()
            btn.setText(directory)

    def find_mech_dir(self, root_dir: str) -> str | None:
        """
        Walks root_dir recursively and returns the first path
        to a subdirectory named 'mech', or None if not found.
        """
        for current_dir, dirnames, filenames in os.walk(root_dir):
            if 'mech' in dirnames:
                return os.path.join(current_dir, 'mech')
        return None

    def init_inp_file(self, mech_dir: str) -> str:
        pattern = os.path.join(mech_dir, '*.inp')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No .inp file found in {mech_dir!r}")
        return matches[0]

    def init_yaml_file(self, mech_dir: str) -> str:
        pattern = os.path.join(mech_dir, '*.yaml')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No .yaml file found in {mech_dir!r}")
        return matches[0]


# --- PyQt UI Class ---
class OptimaSimulatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optima Experiment Simulation Setup")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

    def setup_ui(self):
        self.tabs = QTabWidget()

        # –– Stress State Tab ––
        self.stress_tab = QWidget()
        self.stress_layout = QVBoxLayout()

        # –– Basal State Tab ––
        self.basal_tab = QWidget()
        self.basal_layout = QVBoxLayout()

        self.stress_widget = RangeSimWidget()
        self.basal_widget = RangeSimWidget()
        self.stress_widget.setObjectName("StressTab")
        self.basal_widget.setObjectName("BasalTab")

        # === Run Simulation ===
        self.stress_run_button = QPushButton("Create XML Files")
        self.stress_run_button.clicked.connect(self.create_xmls_stress)
        self.basal_run_button = QPushButton("Create XML Files")
        self.basal_run_button.clicked.connect(self.create_xmls_basal)

        self.exp_4_stress_tab()
        self.stress_layout.addWidget(self.stress_widget)
        self.stress_layout.addWidget(self.stress_run_button)

        self.basal_layout.addWidget(self.basal_widget)
        self.basal_layout.addWidget(self.basal_run_button)

        self.stress_tab.setLayout(self.stress_layout)
        self.tabs.addTab(self.stress_tab, "Stress State")
        self.basal_tab.setLayout(self.basal_layout)
        self.tabs.addTab(self.basal_tab, "Basal State")

        # set the tab widget as our only child
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def exp_4_stress_tab(self):
        # Experiment section (only in Stress State)
        self.stress_layout.addWidget(QLabel("Experiment:"))
        self.exp_file_btn = QPushButton("Choose experiment file")
        self.exp_file_btn.clicked.connect(self.choose_exp_file)
        self.stress_layout.addWidget(self.exp_file_btn)

        # --- Stress type + magnitude inputs ---
        row = QHBoxLayout()

        row.addWidget(QLabel("Stress type:"))
        self.stress_type_cb = QComboBox()
        self.stress_type_cb.addItems(["rapamycin", "starvation"])
        self.stress_type_cb.currentTextChanged.connect(self.on_stress_inp_changed)
        row.addWidget(self.stress_type_cb, stretch=1)

        row.addWidget(QLabel("Level / Conc.:"))
        self.stress_value_le = QLineEdit()
        self.stress_value_le.setPlaceholderText("e.g. 100e-12")
        self.stress_value_le.textChanged.connect(self.on_stress_inp_changed)
        row.addWidget(self.stress_value_le, stretch=1)
        self.stress_layout.addLayout(row)

    def _add_ranges_and_simulation(self, layout: QVBoxLayout):
        """
        Helper to add Theoretical Ranges and Simulation sections
        to whichever tab is calling it.
        """
        # === Theoretical Range Section ===
        layout.addWidget(QLabel("Theoretical Ranges:"))
        self.range_file_btn = QPushButton("Choose range file")
        self.range_file_btn.clicked.connect(self.choose_range_file)
        layout.addWidget(self.range_file_btn)

        self.scaling_input = QLineEdit()
        self.scaling_input.setText("1e-12")
        self.scaling_input.setPlaceholderText("Enter scaling factor (e.g., 1e-12)")
        layout.addWidget(self.scaling_input)

        # === Simulation Section ===

        layout.addWidget(QLabel("Input mechanism:"))
        self.inp_file_btn = QPushButton("Choose input file")
        self.yaml_file_btn = QPushButton("Choose template .yaml file")
        self.yaml_file_btn.clicked.connect(self.choose_yaml_file)
        self.inp_file_btn.clicked.connect(self.choose_input_file)
        layout.addWidget(self.inp_file_btn)
        layout.addWidget(self.yaml_file_btn)

        layout.addWidget(QLabel("Generate:"))
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
        self.num_xml_input.setRange(1, 10000)
        self.num_xml_input.valueChanged.connect(self.on_num_xmls_changed)
        self.num_xml_input.setPrefix("# of XMLs: ")
        layout.addWidget(self.num_xml_input)

        # === Run Simulation ===
        self.run_button = QPushButton("Create XML Files")
        self.run_button.clicked.connect(self.create_xmls)
        layout.addWidget(self.run_button)

    def choose_exp_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Experiment File", "", "Data Files (*.csv *.xlsx)")
        if file:
            self.exp_file_btn.setText(file)

    def on_stress_inp_changed(self):
        pass

    def on_num_xmls_changed(self):
        self.num_xml = self.num_xml_input.value()
        print(f"Number of XMLs: {self.num_xml}\n")

    def stress_parse(self):
        stress_value = self.stress_value_le.text()
        stress_type = self.stress_type_cb.currentText()
        self.stresses = {}
        if stress_type == "rapamycin":
            txt = stress_value.strip()
            print(">>> stress_value_le.text() repr:", repr(txt))
            # val = float(txt)   # ← this is where it blows up

            if not txt:
                QMessageBox.warning(self, "Input Error",
                                    "Please enter a numeric concentration for rapamycin.")
                return
            try:
                val = float(txt)
            except ValueError:
                QMessageBox.critical(self, "Input Error",
                                     f"Cannot parse '{txt}' as a number.")
                return
            self.stresses = {stress_type: (val, "molecular_species")}
        else:  # starvation
            self.stresses = {stress_type: (0.0, "none")}

# Define a function to generate .xml and .opp file contents
    def generate_opp_content(self, xml_folder: str, worksheet_name: str,
                             num_xml: int,
                             mech_file: str = "7_Krisztian/mech/BCRN6.inp",
                             yaml_file: str = "7_Krisztian/mech/BCRN6.yaml",
                             time_limit: int = 50, thread_limit: int = 32,
                             settings_tag: str = "systems_biology",
                             solver: str = "cantera", extension: str = ".xml",
                             ) -> str:
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
        for idx, xml in enumerate(xml_files):
            if idx < num_xml:
                mechtest += f"      NAME {xml.as_posix()}\n"

        mechtest += "END\n"

        return mechmod + "\n" + mechtest

    def create_xmls_stress(self):
        try:
            exp_xlsx_path = self.exp_file_btn.text()
            range_csv = self.stress_widget.range_file_btn.text()  #self.range_file_btn.text()
            xml_template = self.stress_widget.template_file_btn.text()
            output_dir = self.stress_widget.output_dir_btn.text()
            opp_output_dir = self.stress_widget.opp_output_dir_btn.text()
            scaling_factor = float(self.stress_widget.scaling_input.text())
            num_xml = self.stress_widget.num_xml_input.value()

            # Parse stresses
            # --- parse stresses safely ---

            # Read all sheets from Excel
            all_sheets = pd.read_excel(exp_xlsx_path, sheet_name=None)  # dict of {sheet_name: DataFrame}

            # Extract BibTeX from the last sheet
            last_sheet_name = list(all_sheets.keys())[-1]   # Last worksheet should be the BibTeX sheet
            bibtex_df = all_sheets[last_sheet_name]
            # Ha nem lenne header a BibTex-nel, akk ezzel kell beolvasni a sheetet: bibtex_df = pd.read_excel(exp_xlsx_path, sheet_name=last_sheet_name, header=None)

            # Join all non-empty strings from the first column into a BibTeX string
            bibtex_lines = bibtex_df.iloc[:, 0].dropna().astype(str).tolist()
            bibtex_str = "\n".join(bibtex_lines)

            print("\n", bibtex_str, "\n")

            if not bibtex_str.strip():  # If no BibTeX found, raise an error
                raise ValueError(f"No valid BibTeX found in the last worksheet.\n"
                                 f"Extracted string was:\n{bibtex_str!r}")

            date = datetime.datetime.now()
            self.stress_parse()

            for sheet_name in list(all_sheets.keys())[:-1]:
                df = all_sheets[sheet_name]
                exp = Experiment(df, self.stresses, bibtex_str)
                exp.name = sheet_name
                rng = TheoreticalRanges(range_csv, scaling_factor)
                sim = Simulation(rng, exp)

                if sim.omitted:
                    QMessageBox.warning(self, "Missing Ranges",
                        f"The following species were not in your ranges file and will be skipped:\n\n"
                        + ", ".join(sim.omitted))

                sim.create_xml_files(output_dir, num_xml, xml_template)
                mech_file = self.stress_widget.inp_file_btn.text()
                yaml_file = self.stress_widget.yaml_file_btn.text()
                opp_content = self.generate_opp_content(output_dir, sheet_name, num_xml=num_xml, mech_file=mech_file, yaml_file=yaml_file)  # Create .opp file content
                opp_filename = f"{date.year}{date.month}{date.day}_BCRN_{exp.bibtex['author'].split()[0][:-1]}_{sheet_name}.opp" # Define output .opp file path
                with open(os.path.join(opp_output_dir, opp_filename), "w") as f:
                    f.write(opp_content)

            QMessageBox.information(self, "Success", "XML files have been successfully generated for all worksheets.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def create_xmls_basal(self):
        # TODO: Implement basal state XML generation
        pass


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
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/2025517_BCRN_Quan_Bec_LC3.opp
# bin/Release/OptimaPP 7_Krisztian/1_mechtest/2025517_BCRN_CovCor_new_sampling.opp


# bin/Release/OptimaPP Krisztian_Opt/mechtest/2025517_BCRN_CovCor_new_sampling.opp


