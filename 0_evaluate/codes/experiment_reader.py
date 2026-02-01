from __future__ import annotations
import pandas as pd
import bibtexparser


class Experiment:
    def __init__(self, data_source, # data_source: a pandas DataFrame
                 inputs: dict[str, float], species_sigmas: dict[str, float],
                 sheet_name: str, bounds: dict[str, tuple[float, float]],
                 bibtex: str, output_species: list[str], ics_df: pd.DataFrame) -> None:
        self.inputs = inputs
        self.sigmas = species_sigmas
        self.bounds = bounds
        if isinstance(data_source, pd.DataFrame):   # Ha pandas DataFrame (azaz xlsx worksheet)
            self.name = sheet_name
            self.experiment_data = data_source.copy()
        else:
            raise ValueError("data_source must be a pandas DataFrame")

        self.bibtex = self.parse_bibtex(bibtex)
        self.non_species_cols = {"TIME"}
        self.output_species = output_species
        self.ics_df = ics_df

        self.process_data()

    def parse_bibtex(self, bibtex_str):
        parser = bibtexparser.loads(bibtex_str)
        if not parser.entries:
            raise ValueError("No valid BibTeX entry found.")
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
        self.experiment_data.time = self.experiment_data.time #* 60  # Converting hrs to mins - if I convert, some xmls fail for some reaseon
        self.experiment_data = self.experiment_data.dropna()
        self.species = [v for v in self.experiment_data.columns if v.upper() not in self.non_species_cols and "STD" not in v.upper()]

    def quantitated_exp_data(self, ics: dict[str, float]) -> pd.DataFrame:
        quant_Data = self.experiment_data.copy()

        for col in quant_Data.columns:
            if col.upper() not in self.non_species_cols:
                if 'STD' in col.upper():
                    quant_Data[col] *= ics[col[0:-4].upper()]
                else:
                    quant_Data[col] *= ics[col.upper()]
                if 'STD' not in col.upper() and f"{col}_STD" not in quant_Data.columns:
                    # add a new column called f"{col}_STD" filled with sigmas[col.upper()]
                    quant_Data[f"{col}_STD"] = self.sigmas[col.upper()]

        for s in self.output_species:
            if s.upper() not in [col.upper() for col in quant_Data.columns]:
                quant_Data[s] = self.ics_df[self.ics_df['species'] == s].iloc[0, 2]    # "exp_data" = value of the csv column

        for col in quant_Data.columns:
            if col.upper() not in self.non_species_cols:
                if 'STD' not in col.upper() and f"{col}_STD" not in quant_Data.columns:
                    # add a new column called f"{col}_STD" filled with sigmas[col.upper()]
                    quant_Data[f"{col}_STD"] = self.sigmas[col.upper()]

        self.quant_data = quant_Data
        return self.quant_data

    def check_compatibility(self) -> None:
        for s in self.species:
            if s.upper() not in self.bounds.keys():
                self.bounds[s] = [0, 0]
                print(f"Creating new entry for {s}\n")
            #else:
            #    print('All compatible\n')
