"""Helpers for adapting experimental time-course data to Starve/Rap workflows."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd


DEFAULT_NON_SPECIES_COLS = {"TIME"}


def load_experiment_excel(path: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Load an experimental worksheet into a DataFrame."""
    return pd.read_excel(path, sheet_name=sheet_name)


def normalize_experiment_df(
    df: pd.DataFrame,
    non_species_cols: Iterable[str] = DEFAULT_NON_SPECIES_COLS,
) -> pd.DataFrame:
    """Standardize column casing, rename TIME column, and drop missing rows."""
    data = df.copy()
    data.columns = [col.upper() for col in data.columns]
    data.rename(columns={"TIME": "time"}, inplace=True)
    data = data.dropna()
    _ = non_species_cols
    return data


def quantitate_experiment_data(
    df: pd.DataFrame,
    ics: Mapping[str, float],
    sigmas: Mapping[str, float],
    non_species_cols: Iterable[str] = DEFAULT_NON_SPECIES_COLS,
) -> pd.DataFrame:
    """Scale experimental data by ICs and attach standard deviation columns."""
    quant_data = df.copy()
    non_species = {col.upper() for col in non_species_cols}

    for col in list(quant_data.columns):
        if col.upper() in non_species or col.lower() == "time":
            continue

        if col.upper().endswith("_STD"):
            base_name = col[:-4].upper()
            if base_name in ics:
                quant_data[col] *= ics[base_name]
            continue

        species = col.upper()
        if species in ics:
            quant_data[col] *= ics[species]

        std_col = f"{species}_STD"
        if std_col not in quant_data.columns:
            sigma = sigmas.get(species, 2.5e-11)
            quant_data[std_col] = sigma

    return quant_data


def build_experimental_exp_data(
    df: pd.DataFrame,
    ics: Mapping[str, float],
    sigmas: Mapping[str, float],
    non_species_cols: Iterable[str] = DEFAULT_NON_SPECIES_COLS,
) -> pd.DataFrame:
    """Normalize and quantitate experimental data for XML generation."""
    normalized = normalize_experiment_df(df, non_species_cols=non_species_cols)
    return quantitate_experiment_data(
        normalized,
        ics=ics,
        sigmas=sigmas,
        non_species_cols=non_species_cols,
    )


def apply_experimental_data_to_model(
    model,
    df: pd.DataFrame,
    non_species_cols: Iterable[str] = DEFAULT_NON_SPECIES_COLS,
) -> pd.DataFrame:
    """Attach experimental data to a Starve/Rap Model instance.

    Example usage in a notebook:
        exp_df = load_experiment_excel("../input_files/Nitin_rap.xlsx", sheet_name=0)
        model.exp_data = apply_experimental_data_to_model(model, exp_df)
    """
    ics_map = dict(zip(model.ics_df["species"].str.upper(), model.ics_df["value"]))
    model.exp_data = build_experimental_exp_data(
        df,
        ics=ics_map,
        sigmas=model.sigmas,
        non_species_cols=non_species_cols,
    )
    return model.exp_data
