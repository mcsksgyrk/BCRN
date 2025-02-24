import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

class OptimaOutput:
    def __init__(self, job_name: Union[str, Path],
                 optima_path: Optional[Union[str, Path]] = None):

        self.job_name = str(job_name)
        if optima_path == None:
            self.optima_path = Path("/home/szupernikusz/Opt/outputs")
        self.job_folder = self.optima_path / job_name
        file_path = self.job_folder / "mechanismInfo.txt"
        try:
            with open(file_path, "r") as f:
                self.mech_info = f.read()
        except Exception as e:
            raise e


class OptimaMechtest(OptimaOutput):
    def __init__(self, job_name: Union[str, Path],
                 optima_path: Optional[Union[str, Path]] = None,
                 errf_type: Union[str, List[str]] = "default"):
        super().__init__(job_name, optima_path)

        self._errf_files = {
           "default": "errfValues",
           "data_series": "errfValues_by_data_series", 
           "points": "errfValues_by_points",
           "species": "errfValues_by_species"
       }
        if (self.job_folder / "debug").exists():
            self.all_data = {}
            for csv_data in (self.job_folder / "debug").glob("*.csv"):
                try:
                    self.all_data[csv_data.stem] = pd.read_csv(csv_data)
                except Exception as e:
                    print(e)
        
        try:
            if isinstance(errf_type, str):
                file_name = self._errf_files.get(errf_type, "errfValues")
                self.errfValues = pd.read_csv(
                    self.job_folder / file_name,
                    header = 9,
                    delim_whitespace = True
                )
            else:
                self.errfValues = {}
                for e_type in errf_type:
                    file_name = self._errf_files.get(errf_type, "errfValues")
                    self.errfValues[e_type] = pd.read_csv(
                        self.job_folder / file_name,
                        header = 9,
                        delim_whitespace = True
                    )
                
            if (self.job_folder / "sigmas").is_file():
                self.sigmas = pd.read_csv(
                    self.job_folder / "sigmas",
                    skiprows = [0,2],
                    delim_whitespace = True
                )
        except Exception as e:
            raise e


class OptimaSensitivity(OptimaOutput):
    def __init__(self, job_name: Union[str, Path], optima_path: Optional[Union[str, Path]] = None):
        super().__init__(job_name, optima_path)
        try:
            self.reactionList = pd.read_csv(self.job_folder / "reactionList.txt",
                                            delim_whitespace = True
                                            )
            self.sensitivityResults = pd.read_csv(self.job_folder / "sensitivityResults",
                                                  header = 6,
                                                  delim_whitespace = True
                                            )
        except Exception as e:
            raise e
    def _calc_overal_impact(self, sigma, data_m):
        I = sigma * np.sqrt(1/(N_cond*N_tim)*s**2)
        return I
    def calc_normalised_sensitivity(self):
        return

mech = OptimaMechtest("20250123_BCRN_cor.opp")
stac_eq_df = pd.concat({k: v.iloc[-1] for k, v in mech.all_data.items()}, axis=1)
df_basal = stac_eq_df.iloc[3:-1].T
basal_cov = df_basal.cov()
basal_corr = df_basal.corr()
basal_mean = df_basal.mean()
basal_std = df_basal.std()
basal_cov

from scipy import stats
mvd = stats.multivariate_normal(mean=basal_mean, cov=basal_cov)