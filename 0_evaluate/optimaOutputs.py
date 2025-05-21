import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

class OptimaOutput:
    def __init__(self, job_name: Union[str, Path],
                 optima_path: Optional[Union[str, Path]] = None):

        self.job_name = str(job_name)
        if optima_path == None:
            self.optima_path = Path("/home/nvme/Opt/outputs")
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
df_basal = stac_eq_df.iloc[3:-1].T*10e12
basal_cov = df_basal.cov()
basal_corr = df_basal.corr()
basal_corr
import matplotlib.pyplot as plt
from IPython.display import display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 2)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(basal_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.savefig('test')
basal_mean = df_basal.mean()
basal_std = df_basal.std()
eigenvals = np.linalg.eigvals(basal_cov)
eigs = np.linalg.eig(basal_cov)

np.random.multivariate_normal(basal_mean, basal_cov, tol=1, size=(100))

from scipy import stats
mvd = stats.multivariate_normal(mean=basal_mean, cov=basal_cov)
mv_normal = stats.multivariate_normal(mean=basal_mean, cov=basal_cov, allow_singular=True)
mv_normal

arr1 = np.nan_to_num(basal_corr, nan=0.0)
eigenvalues, eigenvectors = np.linalg.eigh(arr1)
# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
names = basal_corr.columns.to_numpy()[idx]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Method 1 - PCA using numpy:")
print("Eigenvalues:", eigenvalues)
print("Explained variance ratio:", explained_variance_ratio)
print("\n")

vec = eigenvectors[:,1]
sorted_indices = np.argsort(np.abs(vec))[::-1]
sorted_vec = vec[sorted_indices]
for idx,x in enumerate(sorted_vec):
    if abs(x) > 0.1: 
        print(names[idx],":",x)