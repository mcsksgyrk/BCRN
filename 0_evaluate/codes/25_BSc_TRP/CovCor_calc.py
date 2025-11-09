import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import matplotlib.pyplot as plt


class OptimaOutput:
    def __init__(self, job_name: Union[str, Path],
                 optima_path: Optional[Union[str, Path]] = None):

        self.job_name = str(job_name)
        if optima_path is None:
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
                 errf_type: Union[str, List[str]] = "default"):     # Union means that it can be either this or that type
        super().__init__(job_name, optima_path)                     # i.e., now either str, or list[str]

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
                    self.all_data[csv_data.stem] = pd.read_csv(csv_data)    # key - value pairs --> key is csv filename without
                                                                            # the extension, value is the .csv as a pd df
                except Exception as e:                                      # i.e., we read all .csv files into a dict (= all_data)
                    print(e)

        if (self.job_folder / "mechTestResults_BCRN6.csv").exists():
            all_sheets_dP = pd.read_csv(self.job_folder / "mechTestResults_BCRN6.csv",  # dP as in the info in dataPoints
                                        header=None,
                                        delimiter=';',
                                        index_col=False,
                                        names=['xml', 'time_point', 'species', 'dP_val', 'sim_val'])
            self.get_the34(all_sheets_dP=all_sheets_dP)

        try:
            if isinstance(errf_type, str):                                  # isinstance(alma, type1) checks if alma is of type 'type1'
                file_name = self._errf_files.get(errf_type, "errfValues")   # i.e., now, it checks if errf_type is a string or not
                self.errfValues = pd.read_csv(                              # since according to the type def. above, it could
                    self.job_folder / file_name,                            # either be a string or a list of strings
                    header = 9,
                    delimiter=r"\s+"                                        # reading the errfValues file as a .csv, omitting the first
                )                                                           # 9 lines of the file (as the header)
            else:
                self.errfValues = {}
                for e_type in errf_type:
                    file_name = self._errf_files.get(errf_type, "errfValues")   # this is a dict() method, such that
                    self.errfValues[e_type] = pd.read_csv(                      # .get(key, default_value) returns the key
                        self.job_folder / file_name,                            # if it's in the dict(), or the default_value
                        header = 9,                                             # if the key is not in the dict()
                        delimiter=r"\s+"
                    )

            if (self.job_folder / "sigmas").is_file():
                self.sigmas = pd.read_csv(
                    self.job_folder / "sigmas",
                    skiprows = [0,2],
                    delimiter=r"\s+"
                )
        except Exception as e:
            raise e

        stac_eq_df = pd.concat({k: v.iloc[-1] for k, v in self.all_data.items()}, axis=1)       # makes a dict() with the
        followed34 = pd.concat({k: v.iloc[-1] for k, v in self.xml_dP.items()}, axis=1)         # same keys, the values is
        self.df_basal = stac_eq_df.iloc[3:-1].T * 1e12                                         # the last element i.e.,
        self.df_followed34 = followed34.T * 1e12                                               # t = last time_point

    def __str__(self):
        if not self.df_basal.empty and not self.df_followed34.empty:
            return f"Mech object was successfully generated with fields\ndf_basal: {self.df_basal.shape}\ndf_followed34: {self.df_followed34.shape}\nsigmas: {self.sigmas.shape}"
        elif self.df_basal.empty and not self.df_followed34.empty:
            return "Unsuccessful, error with df_basal"
        elif not self.df_basal.empty and self.df_followed34.empty:
            return "Unsuccessful, error with df_followed34"
        else:
            return "Mech object was not successfully created"

    def get_the34(self, all_sheets_dP):
        self.xml_dP: dict[str, pd.DataFrame] = {}
        time_point = 0
        species = 'ilyen_species_tuti_nem_lesz'
        for idx, row in all_sheets_dP.iterrows():
            xml_name = row.xml
            sim_val = row.sim_val

            if row.species == species:
                time_point = time_point + 1
            else:
                species = row.species
                time_point = 1
            
            if xml_name not in self.xml_dP.keys():
                self.xml_dP[xml_name] = pd.DataFrame()  # Initialize inner dict
            self.xml_dP[xml_name].loc[time_point, species] = sim_val


class OptimaSensitivity(OptimaOutput):
    def __init__(self, job_name: Union[str, Path], optima_path: Optional[Union[str, Path]] = None):
        super().__init__(job_name, optima_path)
        try:
            self.reactionList = pd.read_csv(self.job_folder / "reactionList.txt",
                                            sep='\s+')
            self.sensitivityResults = pd.read_csv(self.job_folder / "sensitivityResults",
                                                  header = 6,
                                                  sep='\s+')
        except Exception as e:
            raise e
    
    def _calc_overal_impact(self, sigma, data_m):
        I = sigma * np.sqrt(1/(N_cond*N_tim)*s**2)
        return I
    
    def calc_normalised_sensitivity(self):
        return


def main():
    mech = OptimaMechtest("2025522_BCRN_CovCor_old_sampling.opp")

    print(f"dimenziok: (xml_szam, species_szam)\nall:{mech.df_basal.shape}",
          f"\nfollowed:{mech.df_followed34.shape}")

    basal_cov = mech.df_basal.cov()
    followed34_cov = mech.df_followed34.cov()
    basal_corr = mech.df_basal.corr()
    followed34_corr = mech.df_followed34.corr()
    print(followed34_corr)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 2)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(basal_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.savefig('test_all')
    im = ax.imshow(followed34_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.savefig('test_followed34')

    '''basal_mean = df_basal.mean()
    basal_std = df_basal.std()
    eigenvals = np.linalg.eigvals(basal_cov)
    eigs = np.linalg.eig(basal_cov)
    np.random.multivariate_normal(basal_mean, basal_cov, tol=1, size=(100))
    #mvd = stats.multivariate_normal(mean=basal_mean, cov=basal_cov)
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
            print(names[idx],":",x)'''


if __name__ == "__main__":
    main()