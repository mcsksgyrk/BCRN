# %%
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
import os
import datetime
import subprocess
from mech_dot_inp_gen import *
from visualizations import *
from optima import *
import jinja2
import copy

# %%
class Processing:
    def __init__(self, species_txt: str, ) -> None:
        self.species = self.get_species(species_txt)
        #self.observables, self.not_observables = self.get_observables()
        self.observables = ['AMPKA', 'ADP', 'ULKA', 'AMPKA_ATP', 'AMPKA_AMP',
                            'AMP', 'MTORA', 'AMPK', 'ATP', 'GLUCIN',
                            'MTOR', 'ULK', 'GLUCOUT', 'AMPK_AMP', 'AMPK_ADP',
                            'AMPKA_ADP', 'AMPK_ATP']
        self.not_observables = list(set(self.species) - set(self.observables))

    def get_species(self, species_txt: str) -> list[str]:
        species = []
        with open(species_txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            for s in line.strip().split():
                species.append(s)
        print(len(species), species)
        return species

    def get_observables(self) -> tuple[list[str], list[str]]:
        dont_observe = []
        for s in self.species:
            if s == 'Q' or 'LKB1' in s or 'PP2A' in s or 'GLUT1' in s or 'RAP' in s or 'RK' in s or s in ['RHEB', 'PYRUVATE', 'AKT', 'TSCA']:
                dont_observe.append(s)
            elif s == 'FKBP12' or s == 'REF':
                dont_observe.append(s)
        observe = list(set(self.species) - set(dont_observe))
        print(len(observe), observe)
        return observe, dont_observe

# %%
class Model:
    def __init__(self, model_name: str, ics_df_path, # data_source: a pandas DataFrame
                 dont_observe: list[str], time_points: list[float],
                 time_course_vals: Union[dict[str, list[float]], float],
                 food: str, must_be_zero: list[str]=[], uncertainty=False,
                 inputs: dict[str, float]={'GLUCOUT': 6e-6}): # dict[stress_species: init_cc] = [rap extracell.: [mol/cm^3]]
        self.model_name = model_name
        self.ics_df = pd.read_csv(ics_df_path, sep=',')
        self.ics_df['minconc'] = self.ics_df['minconc'].astype(float)
        self.ics_df['maxconc'] = self.ics_df['maxconc'].astype(float)
        self.ics_df['value'] = self.ics_df['value'].astype(float)
        self.ics_df.species = [s.upper() for s in self.ics_df.species]
        self.species = self.ics_df['species'].to_list()
        self.output_species = list(set(self.species) - set(dont_observe))
        self.time_points = time_points
        if isinstance(time_course_vals, float):
            self.time_course_vals = self.gen_time_course_vals(time_course_vals)
        else:
            self.time_course_vals = time_course_vals
        self.food = food
        self.must_be_zero = must_be_zero
        self.uncertainty = uncertainty
        self.inputs = inputs

        self.ranges = self.get_bounds()
        print(f"range values stored in model.ranges of length {len(self.ranges.keys())}")
        self.get_sigmas()
        self.gen_exp_data()

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = dict()
        for _, row in self.ics_df.iterrows():
            if 'REF' not in row.species and row.species not in self.must_be_zero:
                if row.value < 1e-13:
                    lb = 1e-14
                    ub = 1e-13
                elif row.minconc == row.maxconc:
                    lb = row.minconc
                    ub = row.maxconc
                else:
                    if row.value/2 >= row.minconc:
                        lb = (row.value/2)
                    else:
                        lb = row.minconc
                    if row.value*1.5 <= row.maxconc:
                        ub = (row.value*1.5)
                    else:
                        ub = row.maxconc
            bounds[row.species.upper()] = [lb, ub]
        return bounds

    def get_sigmas(self):
        self.sigmas = dict()
        for key, value in self.ranges.items():
            if self.uncertainty:
                if key in self.must_be_zero: # this line might be unnecessary, as I think we need sigma only if the species is an output
                    self.sigmas[key] = 5e-18
                elif key in self.output_species:
                    if value[1] > value[0]:
                        self.sigmas[key] = ((value[1]-value[0])/8)
                    else:
                        self.sigmas[key] = 5e-13
            else:
                self.sigmas[key] = 2.5e-11

    def gen_time_course_vals(self, time_course_vals):
        species_time_course = {}
        for s in self.species:
            species_time_course[s] = np.ones(len(self.time_points)) * time_course_vals
        return species_time_course

    def gen_exp_data(self):
        self.exp_data = pd.DataFrame()
        self.exp_data['time'] = self.time_points
        for s in self.output_species:
            self.exp_data[s] = self.ics_df[self.ics_df['species'] == s].iloc[0, 2] # "exp_data" = value of the csv column

# %%
class Genesis:
    def __init__(self, model: Model, max_digit = 4):
        self.model = model
        self.maxdigit = max_digit

    def create_xml_files(self, output_xmls_path: str, num_of_xmls: int,
                         xml_template_path: str, prefix: str='', kiir=True) -> None:
        if not os.path.exists(output_xmls_path):
            os.makedirs(output_xmls_path)
        self.num_xmls = num_of_xmls
        self.output_dir = output_xmls_path
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(xml_template_path)))
        self.template = env.get_template(os.path.basename(xml_template_path))

        if kiir:
            for i in range(1, num_of_xmls+1):
                np.random.seed(i+1)
                self.random_ics = self.get_random_ics()
                self.make_xml_output(i, output_xmls_path, prefix)

    def get_random_ics(self) -> dict[str, float]:
        random_ics = {}
        if self.model.uncertainty:
            for key in self.model.species:
                if key not in self.model.must_be_zero:
                    random_ics[key] = np.random.uniform(*self.model.ranges[key])
                else:
                    random_ics[key] = 0
        else:
           random_ics = dict(zip(self.model.ics_df['species'], self.model.ics_df['value']))
        for key in self.model.inputs.keys():
            random_ics[key] = self.model.inputs[key]
        random_ics["REF"] = 1.0
        if self.model.food == 'stac':
            random_ics['Q'] = 6e-6
        return random_ics

    def make_xml_output(self, file_index: int, output_xmls_path: str, prefix) -> None:
        dataPoints = [self.compileDataRow(row.values) for _, row in self.model.exp_data.iterrows()]
        output = self.template.render(ics=self.random_ics, relsigmas=self.model.sigmas,
                                      variables=self.model.output_species, dataPoints=dataPoints)
        padded_number = str(file_index).zfill(self.maxdigit)
        filename = f"{prefix}_{padded_number}.xml"
        with open(os.path.join(output_xmls_path, filename), 'w') as f:
            f.write(output)

    def compileDataRow(self, dataPoints):
        meas = "".join(f"<{v}>{{:.4e}}</{v}>" for v in self.model.exp_data.columns)
        return f"<dataPoint>{meas.format(*dataPoints)}</dataPoint>"

# %%
class Simulation:
    def __init__(self, gen: Genesis, xmls_in_one_opp: int) -> None:
      self.gen = gen
      self.get_xml_vec(xmls_in_one_opp)

    def get_xml_vec(self, xmls_in_one_opp) -> None:
      self.xmls = []
      for i in range(1, self.gen.num_xmls, xmls_in_one_opp):
          xml_cnt = np.arange(i, i+xmls_in_one_opp, 1) 
          self.xmls.append(xml_cnt)

    def opp_gen(self, opp_output_dir, opp_name_prefix, kiir, mech_file, food) -> None:
      self.opps = []
      self.indices = []
      self.food = food
      self.mech_file = mech_file
      self.mech_inp_name = self.mech_file.split('/')[-1].split('.inp')[0]
      for num in self.xmls:
          opp_filename = f"{opp_name_prefix}_{num[-1]}.opp"
          self.opps.append(opp_filename)
          self.indices.append(f"{num[-1]}")
          if kiir:
            opp_content = self.generate_opp_content(xml_folder=self.gen.output_dir,
                                                    num_xmls=num,
                                                    mech_file=mech_file,
                                                    name=self.food)
            with open(os.path.join(opp_output_dir, opp_filename), "w") as f:
              f.write(opp_content)

    def generate_opp_content(self, xml_folder: str, num_xmls: Union[list[int], list[list[int]]],
                             mech_file: str, name: str, time_limit: int = 50, thread_limit: int = 32,
                             settings_tag: str = "systems_biology", solver: str = "cantera") -> str:
      mech_name = mech_file.split('/')[-1].split('.inp')[0]
      yaml = mech_file.split('.inp')[0]

      # Create MECHMOD section
      mechmod = f"""MECHMOD
      USE_NAME         {mech_name}
      MECH_FILE        {mech_file}
      COMPILE_{solver} {yaml}.yaml
      END
      """

      # Create MECHTEST section
      mechtest = f"""MECHTEST
      MECHANISM  {mech_name}
      TIME_LIMIT {time_limit}
      THREAD_LIMIT {thread_limit}
      SETTINGS_TAG {settings_tag}
      FALLBACK_TO_DEFAULT_SETTINGS
      PLOTS FALSE

      SOLVER {solver}
      SAVE_STATES      CSV
      """

      # Add each XML file name
      for xml in num_xmls:
          padded_number = str(xml).zfill(self.gen.maxdigit)
          mechtest += f"      NAME {xml_folder}/{name}_{padded_number}.xml\n"

      mechtest += "END\n"

      return mechmod + "\n" + mechtest

    def sim_runner(self, log_location:str = ''):
      self.parent_path = Path.cwd().parents[4] / "OptimaPP"

      if log_location == '':
        for idx, opp_file in enumerate(self.opps):
            command = ["bin/Release/OptimaPP", f"../Projects/Semmelweis_BCRN/BCRN/1_mechtest/{opp_file}"]
            print(f"Running: {' '.join(command)}")
            subprocess.run(command, check=True, cwd=self.parent_path)
      else:
        for idx, opp_file in enumerate(self.opps):
          command = ["bin/Release/OptimaPP", f"../Projects/Semmelweis_BCRN/BCRN/1_mechtest/{opp_file}"]
          print(f"Running: {' '.join(command)}")
          log_idx = self.xmls[idx][-1]
          with open(f"{log_location}/run_log_stac_starve_rap_{self.food}_{log_idx}.txt", "w") as log:
              subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=self.parent_path)

# %%
class Natural_Selection:
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.sim_data: dict[str, OptimaMechtest] = {}
        self.get_sim_data()
        #self.survival_of_the_fittest()

    def get_sim_data(self) -> None:
        for idx, key in enumerate(self.sim.indices):
            self.sim_data[key] = OptimaMechtest(job_name=self.sim.opps[idx],
                                                input_mech=self.sim.mech_inp_name)

    def sigma_range(self, meas, sim, sigma):
        radius = (sim-meas)/sigma
        return radius
    
    def isit_init(self, row):
        lut = self.sim.gen.model.ics_df
        rel_sigmas = self.sim.gen.model.sigmas
        for k, v in row.items():
            if k in self.condition_vars:
                right_row = lut[lut['species'] == k]

                meas = right_row['value'].iloc[0]

                radius = self.sigma_range(meas=meas, sim=v, sigma=rel_sigmas[k])

                if radius >= 4:
                    self.wrongdoers[k] += 1
                    return False

        return True
    
    def survival_of_the_fittest(self) -> None:
        self.good_xmls = []
        self.get_condition_vars()
        self.wrongdoers = {k: 0 for k in self.condition_vars}
        for idx in self.sim.indices:
            for xml_name, row in self.sim_data[idx].df_followed18.iterrows():
                all_ok = self.isit_init(row)
                if all_ok:
                    self.good_xmls.append(xml_name)
        print(f"Found {len(self.good_xmls)} good xmls")

    def get_condition_vars(self) -> None:
        self.condition_vars = ['RHEBA', 'AKTA', 'TSC', 'MTORA', 'ATP', 'ULK', 'GLUCIN']
# too big LKB1 activity could mean upregged AMPK biologically speaking
# ADP, AMP make complexes just like ATP, but for AMP/ADP the complex concentrations are comparable to the
# cc. of either ADP or AMP --> complex-bound ADP/AMP would throw the calculation off
# LKB1 and PP2A form complexes with the AMPKs --> their cc. should change --> not adequate conditions
    def filtering(self) -> None:
        data = copy.deepcopy(self.sim_data)
        first = True
        self.filtered_basal = pd.DataFrame()
        self.filtered_followed = pd.DataFrame()
        for k, v in data.items():
            v.df_basal.index = v.df_basal.index.str[7:-9]
            v.df_basal = v.df_basal.sort_index()
            if first:
                self.filtered_basal = v.df_basal[[xml in self.good_xmls for xml in v.df_basal.index]]
                self.filtered_followed = v.df_followed18[[xml in self.good_xmls for xml in v.df_followed18.index]]
                first = False
            else:
                self.filtered_basal = pd.concat([self.filtered_basal, v.df_basal[[xml in self.good_xmls for xml in v.df_basal.index]]],
                                        ignore_index=False)
                self.filtered_followed = pd.concat([self.filtered_followed, v.df_followed18[[xml in self.good_xmls for xml in v.df_followed18.index]]],
                                            ignore_index=False)

    def get_cov_cor(self, corr_xmls, keys: list[str]) -> None:
        self.dict_b = {}
        self.dict_f = {}
        self.dict_b_corr = {}
        self.dict_f_corr = {}
        self.dict_b_cov = {}
        self.dict_f_cov = {}
        for idx, alma in enumerate(corr_xmls):
            self.dict_f[f"{keys[idx]}"] = self.filtered_followed.iloc[alma].copy()
            #self.dict_b[f"{keys[idx]}"] = self.filtered_basal.iloc[alma].copy()
            #self.dict_b_corr[f"{keys[idx]}"] = self.filtered_basal.iloc[alma].copy().corr()
            self.dict_f_corr[f"{keys[idx]}"] = self.filtered_followed.iloc[alma].copy().corr()
            self.dict_f_cov[f"{keys[idx]}"] = self.filtered_followed.iloc[alma].copy().cov()
            #self.dict_b_cov[f"{keys[idx]}"] = self.filtered_basal.iloc[alma].copy().cov()

# %%
def plot_correlation_distance_heatmap(dict, corr_xmls, title: str, method='fro',
                                      save_fig=False, nat_select:Natural_Selection=None):
    num_corrs = len(corr_xmls)
    distance_matrix = np.zeros((num_corrs, num_corrs))

    for i, n1 in enumerate(corr_xmls):
        for j, n2 in enumerate(corr_xmls):
            mat1 = dict[str(n1)].values
            mat2 = dict[str(n2)].values
            if method == 'kl':
                mu1 = nat_select.dict_f[str(n1)].mean()
                mu2 = nat_select.dict_f[str(n2)].mean()
                distance_matrix[i, j] = compute_matrix_distance(mat1, mat2, method=method, mu1=mu1, mu2=mu2)
            else:
                distance_matrix[i, j] = compute_matrix_distance(mat1, mat2, method=method)
    plot_it(distance_matrix, corr_xmls, corr_xmls, method, save_fig, title)

def plot_dual_metric_heatmap(dict, keys,
                              method_lower='cmd', method_upper='frobenius',
                              title='Dual Distance Plot',
                              cmap_lower='Blues', cmap_upper='Reds',
                              save_fig=False,
                              nat_select_dict: dict[str, pd.DataFrame] = None):
    n = len(keys)
    data_lower = np.zeros((n, n))
    data_upper = np.zeros((n, n))

    # Compute both metrics for all pairs
    for i, n1 in enumerate(keys):
        for j, n2 in enumerate(keys):
            mat1 = dict[str(n1)]
            mat2 = dict[str(n2)]
            if method_lower == 'kl' or method_upper == 'kl':
                mu1 = nat_select_dict[str(n1)].mean()
                mu2 = nat_select_dict[str(n2)].mean()
            if i >= j:
                data_lower[i, j] = compute_matrix_distance(mat1, mat2, method_lower, mu1=mu1, mu2=mu2)
            elif i < j:
                data_upper[i, j] = compute_matrix_distance(mat1, mat2, method_upper, mu1=mu1, mu2=mu2)

    # Create masks
    mask_lower = np.triu(np.ones_like(data_lower, dtype=bool), k=1)
    mask_upper = np.tril(np.ones_like(data_upper, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot lower triangle
    norm_lower = Normalize(vmin=np.min(data_lower[np.tril_indices(n, -1)]),
                           vmax=np.max(data_lower[np.tril_indices(n, -1)]))
    sns.heatmap(data_lower,
                mask=mask_lower,
                cmap=cmap_lower,
                annot=True,
                square=True,
                linewidths=0.5,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=ax)

    plt.xlabel('Number of XMLs')
    plt.ylabel('Number of XMLs')

    # Plot upper triangle
    norm_upper = Normalize(vmin=np.min(data_upper[np.triu_indices(n, 1)]),
                           vmax=np.max(data_upper[np.triu_indices(n, 1)]))
    sns.heatmap(data_upper,
                mask=mask_upper,
                cmap=cmap_upper,
                annot=True,
                square=True,
                linewidths=0.5,
                cbar=False,
                xticklabels=keys,
                yticklabels=keys,
                ax=ax)

    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(keys, rotation=0, fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)

    # Add both colorbars
    cbar_ax1 = fig.add_axes([0.84, 0.25, 0.02, 0.5])  # Right colorbar
    sm1 = cm.ScalarMappable(cmap=cmap_upper, norm=norm_upper)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label(f'{method_upper.upper()} (upper)', rotation=270, labelpad=15)

    cbar_ax2 = fig.add_axes([0.0, 0.25, 0.02, 0.5])  # Left colorbar
    sm2 = cm.ScalarMappable(cmap=cmap_lower, norm=norm_lower)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, cax=cbar_ax2)
    cbar2.set_label(f'{method_lower.upper()} (lower)', rotation=90, labelpad=15)

    #fig.patch.set_facecolor('#f8f6f6')
    #plt.tight_layout(rect=[0.1, 0, 0.9, 1])

    if save_fig:
        plt.savefig(f"pics/{title.replace(' ', '_')}.png", dpi=300)
        plt.savefig(f"pics/{title.replace(' ', '_')}.pdf")

    plt.show()

def plot_correlation_distance_heatmap_leodit(dict, corr_xmls, title: str, method='frobenius',
                                      save_fig=False, nat_select:Natural_Selection=None):
    num_corrs = len(corr_xmls)
    distance_matrix = np.zeros((num_corrs, num_corrs))

    for i, n1 in enumerate(corr_xmls):
        for j, n2 in enumerate(corr_xmls):
            mat1 = dict[str(n1)]
            mat2 = dict[str(n2)]
            if method == 'kl':
                mu1 = nat_select.dict_b[str(n1)].mean()
                mu2 = nat_select.dict_b[str(n2)].mean()
                distance_matrix[i, j] = compute_matrix_distance(mat1, mat2, method=method, mu1=mu1, mu2=mu2)
            else:
                distance_matrix[i, j] = compute_matrix_distance(mat1, mat2, method=method)
    plot_it(distance_matrix, corr_xmls, corr_xmls, method, save_fig, title)

# %%
proc = Processing('../input_files/species.txt')
print(len(proc.observables), proc.observables)

# %%
#gluc_rap = pd.read_csv('../input_files/gluc_rap_ranges.csv')
#combined = pd.read_csv('../input_files/combined_ranges.csv')
#
#cols = ["minconc", "maxconc", "value"]
#combined[cols] *= 1e-12
#
#combined.to_csv('../input_files/combined_ranges_scaled.csv', index=False)

# %%
t = np.arange(0, 5070, 30)
food = 'stac'
if food == 'starve':
    exp_food = 1e-6
elif food == 'stac':
    exp_food = 6e-6
else:
    raise Exception('food must either be "starve" or "stac"')

# %%
exp_foods = ['stac', 'starve']
must_be_zero = ['RAP', 'RK', 'RKMTORA', 'RAPOUT']
date = datetime.datetime.now()
date_prefix = "2026210"#f"{date.year}{date.month}{date.day}"
t = np.arange(0, 5070, 30)
sims: dict[str, Simulation] = {}

for food in exp_foods:
    if food == 'starve':
        exp_food = 1e-6
    elif food == 'stac':
        exp_food = 6e-6
    else:
        raise Exception('food must either be "starve" or "stac"')

    mod = Model(model_name=food, ics_df_path='../input_files/species_ics_ranges.csv',
                dont_observe=proc.not_observables, time_points=t, time_course_vals=7e-10,
                uncertainty=True, must_be_zero=must_be_zero, inputs={'GLUCOUT': exp_food},
                food=food)

    gen = Genesis(mod, max_digit=4)
    gen.create_xml_files(output_xmls_path=f'/home/szupernikusz/Projects/Semmelweis_BCRN/BCRN/xml/testing_starve_{food}',
                         num_of_xmls=100,
                         xml_template_path='../input_files/std_xml_template.xml',
                         prefix=food, kiir=True)

    sim = Simulation(gen, 100)
    sim.opp_gen(opp_output_dir='/home/szupernikusz/Projects/Semmelweis_BCRN/BCRN/1_mechtest',
                opp_name_prefix=f'{date_prefix}_testing_starve_{food}',
                kiir=True, mech_file=f'/home/szupernikusz/Projects/Semmelweis_BCRN/BCRN/mech/Minimal_gluc_rap_mTOR_v1.inp',
                food=food)

    #sim.sim_runner('/home/szupernikusz/Projects/Semmelweis_BCRN/BCRN/logs/testing')

    sims[food] = sim

# %%
mechs: dict[str, OptimaMechtest] = {}
for k in sims.keys():
    mechs[k] = OptimaMechtest(sims[k].opps[0], sims[k].mech_inp_name)

# %%
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, logm
import seaborn as sns


def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)


def kl_divergence_gaussians(mu0, cov0, mu1, cov1):
    """
    KL divergence D_KL(N0 || N1) between two multivariate Gaussians N0 ~ (mu0, cov0) and N1 ~ (mu1, cov1)
    """
    k = mu0.shape[0]
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0

    term1 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
    term2 = np.trace(cov1_inv @ cov0)
    term3 = diff.T @ cov1_inv @ diff

    return 0.5 * (term1 - k + term2 + term3)


def plot_corr_matrix(df_corr, title="", figsize=(10, 10), vmin=-1, vmax=1, fontsize=7,
                     dropNa = False, save_fig=False, mask_it=False, show_diag=True):

    if dropNa:
        df_corr = df_corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

    mask = None
    if mask_it:
        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=0 if not show_diag else 1)

    fig = plt.figure(figsize=figsize)

    sns.heatmap(df_corr, 
                annot=False,
                mask=mask, 
                cmap='coolwarm', 
                center=0,
                vmin=vmin,
                vmax=vmax,
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=0.3,
                cbar_kws={"shrink": 0.8})
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.title(title)
    plt.tight_layout()
    #fig1.patch.set_facecolor('#f8f6f6')
    if save_fig:
        plt.savefig(f"pics/{title}.pdf")
    plt.show()


def plot_corr_matrix_GPT(df_corr, title="", figsize=(10, 10), vmin=-1, vmax=1, fontsize=7,
                     dropNa=False, save_fig=False, mask_it=False, show_diag=True):

    if dropNa:
        df_corr = df_corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

    mask = None
    if mask_it:
        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=0 if not show_diag else 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.05, 0.05, 0.75, 0.9])  # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0, 0.1, 0.03, 0.8])  # [left, bottom, width, height] for colorbar

    sns.heatmap(df_corr, 
                ax=ax,
                annot=False,
                mask=mask, 
                cmap='coolwarm', 
                center=0,
                vmin=vmin,
                vmax=vmax,
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=0.3,
                cbar_ax=cbar_ax,
                cbar_kws={"shrink": 1})

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
    ax.set_title(title)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"pics/{title}.pdf")

    plt.show()


def correlation_matrix_distance(A, B):
    """Correlation Matrix Distance (CMD), Herdin et al. (2005), https://doi.org/10.1049/el:20057319"""
    numerator = np.trace(A @ B)
    denominator = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - (numerator / denominator)


def rv_coefficient(A, B):
    """RV Coefficient, Escoufier (1973), https://doi.org/10.1016/0378-3758(78)90121-8"""
    numerator = np.trace(A @ B)
    denominator = np.sqrt(np.trace(A @ A) * np.trace(B @ B))
    return numerator / denominator


def safe_airm(A, B, epsilon=1e+1):
    """If COV() matrix is singular, https://doi.org/10.1016/S0047-259X(03)00096-4"""
    A_spd = A + epsilon * np.eye(A.shape[0])
    B_spd = B + epsilon * np.eye(B.shape[0])
    A_inv_sqrt = np.linalg.inv(sqrtm(A_spd))
    log_term = logm(A_inv_sqrt @ B_spd @ A_inv_sqrt)
    return np.linalg.norm(log_term, 'fro')


def affine_invariant_riemannian(A, B):
    """Affine-Invariant Riemannian Distance, https://doi.org/10.1007/s11263-005-3222-z"""
    try:
        A_inv_sqrt = np.linalg.inv(sqrtm(A))
        log_term = logm(A_inv_sqrt @ B @ A_inv_sqrt)
        return np.linalg.norm(log_term, 'fro')
    except:
        print("Matrix is singular, switching to linear shrinkage estimator formula")
        safe_airm(A, B)


def bures_wasserstein_distance(A, B):
    """Bures–Wasserstein Distance, https://doi.org/10.1016/0047-259X(82)90077-X"""
    sqrtA = sqrtm(A)
    middle = sqrtm(sqrtA @ B @ sqrtA)
    return np.sqrt(np.trace(A + B - 2 * middle))


def root_stein_divergence(A, B):
    """Root Stein Divergence, Sra, A New Metric on the Manifold of Kernel Matrices with Application to Matrix Geometric Means, NIPS, 2012"""
    sqrt_product = sqrtm(A @ B)
    return np.trace(A + B - 2 * sqrt_product)


def compute_matrix_distance(mat1, mat2, method='fro', mu1=None, mu2=None):
    if method == 'fro':
        return np.linalg.norm(mat1 - mat2, ord='fro')
    elif method == 'cmd':
        return correlation_matrix_distance(mat1, mat2)
    elif method == 'rv':
        return rv_coefficient(mat1, mat2)
    elif method == 'airm':
        return affine_invariant_riemannian(mat1, mat2)
    elif method == 'bures':
        return bures_wasserstein_distance(mat1, mat2)
    elif method == 'rsd':
        return root_stein_divergence(mat1, mat2)
    elif method == 'kl':
        return kl_mvn(mu1, mat1, mu2, mat2)
    else:
        raise ValueError("Unknown method. Use 'frobenius', 'cmd', 'rv', 'airm', 'skl', 'bures', or 'rsd'.")


def plot_it(distance_matrix, corr_xmls1, corr_xmls2, method, save_fig, title):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix,
                xticklabels=corr_xmls1,
                yticklabels=corr_xmls2,
                annot=True,
                cmap='coolwarm',
                square=True,
                cbar_kws={'label': f'{method.upper()} Distance'})
    plt.title(title)
    plt.xlabel('Number of XMLs - Narrow')
    plt.ylabel('Number of XMLs - Wide')
    plt.tight_layout()
    #fig.patch.set_facecolor('#f8f6f6')
    if save_fig:
        plt.savefig(f"pics/{title}.pdf")
        plt.savefig(f"pics/{title}.png")
    plt.show()


def plot_correlation_matrix_triangle(corr_matrix: np.ndarray, labels=None, title="", 
                                     figsize=(12, 10), cmap='coolwarm', fmt=".2f", 
                                     show_diag=True, save_path=None):
    """
    Plot a lower triangular heatmap of a correlation matrix with annotations.
    """

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0 if not show_diag else 1)

    # Setup the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.85, 'label': 'Correlation'},
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def _normalize_species(species):
    if isinstance(species, str):
        return (species,)
    return tuple(species)


def _stack_runs(df_dict, species, n_expected=None):
    species = _normalize_species(species)
    max_idx = max(df.index.max() for df in df_dict.values())
    if n_expected is not None:
        max_idx = max(max_idx, n_expected)
    idx = pd.RangeIndex(1, max_idx + 1)

    mats = []
    for _, df in df_dict.items():
        tmp = (df.sort_index()
                 .reindex(idx)
                 .apply(pd.to_numeric, errors='coerce')
                 .ffill().bfill())
        if set(species).issubset(tmp.columns):
            mats.append(tmp.loc[:, list(species)].to_numpy())
    if not mats:
        raise ValueError(f"No runs contained the requested species: {species}")
    return idx, np.stack(mats, axis=0)


def _summary(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q10 = np.quantile(X, 0.10, axis=0)
    q25 = np.quantile(X, 0.25, axis=0)
    med = np.quantile(X, 0.50, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    q90 = np.quantile(X, 0.90, axis=0)
    mean = X.mean(axis=0)
    return q10, q25, med, q75, q90, mean


def _spread(y_values, min_gap):
    """
    Given a list of y positions, return adjusted positions so that
    consecutive labels are at least `min_gap` apart (in data units),
    while preserving ordering as much as possible.
    """
    y = np.array(y_values, dtype=float)
    order = np.argsort(y)
    y_sorted = y[order]

    adj = np.empty_like(y_sorted)
    prev = -np.inf
    for i, val in enumerate(y_sorted):
        if val <= prev + min_gap:
            adj[i] = prev + min_gap
        else:
            adj[i] = val
        prev = adj[i]

    result = np.empty_like(adj)
    result[order] = adj
    return result


def plot_compare_summary_gluc(
    df_dict_a: Dict[str, pd.DataFrame],
    df_dict_b: Dict[str, pd.DataFrame],
    label_a: str = 'cond A',
    label_b: str = 'cond B',
    species=('ATP','ADP','AMP'),
    dt_minutes=30,
    total_hours=84,
    show_iqr: bool = True,
    show_p80: bool = False,
    show_mean: bool = True,
    show_median: bool = False,
    sample_trajs: int = 0,
    traj_alpha: float = 0.10,
    band_alpha_iqr: float = 0.20,
    band_alpha_p80: float = 0.10,
    legend_pos: str = "below",
    show_axp: bool = True,
    y_label = r'concentration $[\frac{mol}{cm^{3}}]$',
    figsize=(12, 8),
    curve_labels=False
):
    species = _normalize_species(species)
    n_expected = int(total_hours*60/dt_minutes + 1)

    idx_a, Xa = _stack_runs(df_dict_a, species, n_expected)
    idx_b, Xb = _stack_runs(df_dict_b, species, n_expected)
    t_h_a = (idx_a.values - 1) * dt_minutes / 60.0
    t_h_b = (idx_b.values - 1) * dt_minutes / 60.0

    qa10, qa25, qamed, qa75, qa90, qamean = _summary(Xa)
    qb10, qb25, qbmed, qb75, qb90, qbmean = _summary(Xb)

    fig, axes = plt.subplots(1, len(species), figsize=figsize, sharex=True)
    if len(species) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    samp_a = rng.choice(Xa.shape[0], size=min(sample_trajs, Xa.shape[0]),
                        replace=False) if sample_trajs else []
    samp_b = rng.choice(Xb.shape[0], size=min(sample_trajs, Xb.shape[0]),
                        replace=False) if sample_trajs else []

    for si, sp in enumerate(species):
        ax = axes[si]

        for ridx in samp_a:
            ax.plot(t_h_a, Xa[ridx, :, si], lw=0.6, alpha=traj_alpha)
        for ridx in samp_b:
            ax.plot(t_h_b, Xb[ridx, :, si], lw=0.6, alpha=traj_alpha)

        if show_p80:
            ax.fill_between(t_h_a, qa10[:, si], qa90[:, si],
                            alpha=band_alpha_p80, linewidth=0, label=f'{label_a} 10–90%')
        if show_iqr:
            ax.fill_between(t_h_a, qa25[:, si], qa75[:, si],
                            alpha=band_alpha_iqr, linewidth=0, label=f'{label_a} IQR')

        if show_p80:
            ax.fill_between(t_h_b, qb10[:, si], qb90[:, si],
                            alpha=band_alpha_p80, linewidth=0, label=f'{label_b} 10–90%')
        if show_iqr:
            ax.fill_between(t_h_b, qb25[:, si], qb75[:, si],
                            alpha=band_alpha_iqr, linewidth=0, label=f'{label_b} IQR')

        if curve_labels:
            y0_a = qamean[0, si]
            y0_b = qbmean[0, si]
            ax.text(-2.5, y0_a, f"{label_a}", color='tab:blue', va='center', ha='right',
                    fontsize=12, weight='bold')
            ax.text(-2.5, y0_b, f"{label_b}", color='tab:orange', va='center', ha='right',
                    fontsize=12, weight='bold')

        if show_mean:
            ax.plot(t_h_a, qamean[:, si], lw=2, label=f'{label_a} mean')
            ax.plot(t_h_b, qbmean[:, si], lw=2, label=f'{label_b} mean')
        if show_median:
            ax.plot(t_h_a, qamed[:, si], lw=1, linestyle='--', label=f'{label_a} median')
            ax.plot(t_h_b, qbmed[:, si], lw=1, linestyle='--', label=f'{label_b} median')

        ax.set_title("Extracellular Glucose Concentrations - Simulations and Measurements",
                     fontsize=22)
        ax.set_xlim(-0.2, total_hours)
        ax.set_xticks(np.arange(0, total_hours+0.1, 6))
        ax.tick_params(axis='x', labelsize=18)
        if si == 0:
            ax.set_ylabel(y_label, fontsize=22)
        ax.set_xlabel('time [h]', fontsize=22)
        ax.grid(alpha=0.2, linestyle=':')
        ax.tick_params(axis='y', labelsize=18)

        # ------- EXP DATA + SQUARED ERROR QUANTIFICATION (GLUCOUT only) -------
        if sp == 'GLUCOUT' and (label_a == 'Starved' or label_b == 'Starved'):
            sejeong_gluc = pd.read_csv('../input_files/sejeong_gluc_cc_starve.csv')
            sejeong_gluc.t = sejeong_gluc.t
            sejeong_gluc.y = sejeong_gluc.y #* 1e-6
            sejeong_gluc.sigma = sejeong_gluc.sigma #* 1e-6
            # experimental data
            ax.errorbar(
                sejeong_gluc.t,
                sejeong_gluc.y,
                yerr=sejeong_gluc.sigma,
                fmt='x',
                capsize=5,
                lw=1.8,
                markersize=10,
                color='tab:red',
                ecolor='tab:red',
                elinewidth=2,
                alpha=0.9,
                label='Experimental Data'
            )

            # squared errors between experimental data and sim mean (cond A)
            mean_sim = qamean[:, si]                            # mean curve for GLUCOUT
            sim_at_meas = np.interp(sejeong_gluc.t, t_h_a, mean_sim)
            sq_errors = (sim_at_meas - sejeong_gluc.y) ** 2
            sse = sq_errors.sum()
            mse = sq_errors.mean()
            rmse = np.sqrt(mse)

            # print to console (optional)
            print(f"[GLUCOUT] SSE = {sse:.4e}, RMSE = {rmse:.4e}")

            # show on the plot
            #ax.text(
            #    0.98, 0.82,
            #    r"$\sum (y_{\mathrm{sim}}-y_{\mathrm{exp}})^2$"
            #    + f" = {sse:.2e}\nRMSE = {rmse:.2e}",
            #    transform=ax.transAxes,
            #    ha='right', va='bottom',
            #    fontsize=20,
            #    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
            #)
        # ----------------------------------------------------------------------

    if show_axp and set(species) == {'ATP','ADP','AMP'}:
        axes[0].plot(t_h_a, qamean.sum(axis=1), lw=1.2, linestyle=':',
                     label=f'AXP mean – {label_a}')
        axes[0].plot(t_h_b, qbmean.sum(axis=1), lw=1.2, linestyle=':',
                     label=f'AXP mean – {label_b}')

    handles, labels = axes[0].get_legend_handles_labels()
    if legend_pos == "below":
        fig.legend(handles, labels, frameon=False, loc='lower center',
                   bbox_to_anchor=(0.5, -0.01), ncol=6, fontsize=20)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    elif legend_pos == "above":
        fig.legend(handles, labels, frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, 1.15), ncol=6, fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        axes[-1].legend(handles, labels, frameon=False, loc='upper right', fontsize=20)
        fig.tight_layout()

    fig.patch.set_facecolor('white')
    return fig


def plot_compare_summary(
    df_dict_a: Dict[str, pd.DataFrame],
    df_dict_b: Dict[str, pd.DataFrame],
    label_a: str = 'cond A',
    label_b: str = 'cond B',
    species=('ATP','ADP','AMP'),
    dt_minutes=30,
    total_hours=84,
    # ---- layout switch ----
    layout_mode: str = "species",   # "species" | "condition"
    # ---- stat controls ----
    show_iqr: bool = True,
    show_p80: bool = False,
    show_mean: bool = True,
    show_median: bool = False,
    sample_trajs: int = 0,          # 0 -> no thin trajectories
    traj_alpha: float = 0.10,
    band_alpha_iqr: float = 0.20,
    band_alpha_p80: float = 0.10,
    legend_pos: str = "below",      # "below" | "above" | "inside"
    # -----------------------
    show_axp: bool = True,          # only used if plotting the AXP trio
    figsize=(12, 8),
    curve_labels: bool = False,
    curve_label_min_gap_frac: float = 0.02,
    curve_label_xpad: float = -3.0,
    curve_label_connectors: bool = True,
    # ---- NEW: separate figures per species ----
    multi_fig: bool = False,
):
    """
    Compare conditions A vs B for the given species.

    layout_mode = "species":
        One subplot per species. Each subplot shows both conditions.

    layout_mode = "condition":
        Two subplots (A, B). Each subplot shows all species for that condition.

    multi_fig = True:
        Instead of one figure for all species, return a *list of figures*,
        one figure per species, using the chosen layout_mode.
    """
    if layout_mode not in {"species", "condition"}:
        raise ValueError('layout_mode must be "species" or "condition".')

    species = _normalize_species(species)

    # --- NEW: create one fig per species if requested -------------------
    if multi_fig and len(species) > 1:
        figs = []
        for sp in species:
            figs.append(
                plot_compare_summary(
                    df_dict_a=df_dict_a,
                    df_dict_b=df_dict_b,
                    label_a=label_a,
                    label_b=label_b,
                    species=(sp,),
                    dt_minutes=dt_minutes,
                    total_hours=total_hours,
                    layout_mode=layout_mode,
                    show_iqr=show_iqr,
                    show_p80=show_p80,
                    show_mean=show_mean,
                    show_median=show_median,
                    sample_trajs=sample_trajs,
                    traj_alpha=traj_alpha,
                    band_alpha_iqr=band_alpha_iqr,
                    band_alpha_p80=band_alpha_p80,
                    legend_pos=legend_pos,
                    show_axp=show_axp,
                    figsize=figsize,
                    curve_labels=curve_labels,
                    curve_label_min_gap_frac=curve_label_min_gap_frac,
                    curve_label_xpad=curve_label_xpad,
                    curve_label_connectors=curve_label_connectors,
                    multi_fig=False,      # avoid infinite recursion
                )
            )
        return figs
    # --------------------------------------------------------------------

    n_expected = int(total_hours*60/dt_minutes + 1)

    # stack & summarize
    idx_a, Xa = _stack_runs(df_dict_a, species, n_expected)  # [runs, time, species]
    idx_b, Xb = _stack_runs(df_dict_b, species, n_expected)
    t_h_a = (idx_a.values - 1) * dt_minutes / 60.0
    t_h_b = (idx_b.values - 1) * dt_minutes / 60.0

    qa10, qa25, qamed, qa75, qa90, qamean = _summary(Xa)
    qb10, qb25, qbmed, qb75, qb90, qbmean = _summary(Xb)

    rng = np.random.default_rng(0)
    samp_a = rng.choice(Xa.shape[0], size=min(sample_trajs, Xa.shape[0]), replace=False) if sample_trajs else []
    samp_b = rng.choice(Xb.shape[0], size=min(sample_trajs, Xb.shape[0]), replace=False) if sample_trajs else []

    legend_data = {label_a: {}, label_b: {}}

    # ---------------- LAYOUT 1: one subplot per species ----------------
    if layout_mode == "species":
        fig, axes = plt.subplots(1, len(species), figsize=figsize, sharex=True)
        if len(species) == 1:
            axes = [axes]

        for si, sp in enumerate(species):
            ax = axes[si]

            for ridx in samp_a:
                ax.plot(t_h_a, Xa[ridx, :, si], lw=0.6, alpha=traj_alpha)
            for ridx in samp_b:
                ax.plot(t_h_b, Xb[ridx, :, si], lw=0.6, alpha=traj_alpha)

            if show_p80:
                ax.fill_between(t_h_a, qa10[:, si], qa90[:, si],
                                alpha=band_alpha_p80, linewidth=0, label=f'{label_a} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_a, qa25[:, si], qa75[:, si],
                                alpha=band_alpha_iqr, linewidth=0, label=f'{label_a} IQR')

            if show_p80:
                ax.fill_between(t_h_b, qb10[:, si], qb90[:, si],
                                alpha=band_alpha_p80, linewidth=0, label=f'{label_b} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_b, qb25[:, si], qb75[:, si],
                                alpha=band_alpha_iqr, linewidth=0, label=f'{label_b} IQR')

            if curve_labels:
                y0_a = qamean[0, si]
                y0_b = qbmean[0, si]
                ax.text(-2.5, y0_a, f"{label_a}", color='tab:blue',
                        va='center', ha='right', fontsize=12, weight='bold')
                ax.text(-2.5, y0_b, f"{label_b}", color='tab:orange',
                        va='center', ha='right', fontsize=12, weight='bold')

            if show_mean:
                ax.plot(t_h_a, qamean[:, si], lw=2, label=f'{label_a} mean')
                ax.plot(t_h_b, qbmean[:, si], lw=2, label=f'{label_b} mean')
            if show_median:
                ax.plot(t_h_a, qamed[:, si], lw=1, linestyle='--', label=f'{label_a} median')
                ax.plot(t_h_b, qbmed[:, si], lw=1, linestyle='--', label=f'{label_b} median')

            ax.set_title(f"Time Profile of {sp} Dynamics for N = {Xa.shape[0]} Simulations", fontsize=20)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            if si == 0:
                ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
            ax.set_xlabel('time [h]', fontsize=12)
            ax.grid(alpha=0.2, linestyle=':')

            if sp == 'GLUCOUT' and (label_a == 'Starved' or label_b == 'Starved'):
                sejeong_gluc = pd.read_csv('../input_files/sejeong_gluc_cc_starve.csv')
                sejeong_gluc.t = sejeong_gluc.t
                sejeong_gluc.y = sejeong_gluc.y #* 1e-6
                sejeong_gluc.sigma = sejeong_gluc.sigma #* 1e-6
                ax.errorbar(
                    sejeong_gluc.t,
                    sejeong_gluc.y,
                    yerr=sejeong_gluc.sigma,
                    fmt='x',
                    capsize=5,
                    lw=1.8,
                    markersize=5,
                    color='tab:red',
                    ecolor='tab:red',
                    elinewidth=2,
                    alpha=0.9
                )

    # ---------------- LAYOUT 2: one subplot per condition ----------------
    else: # stac vs starve
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

        def _plot_condition_panel(ax, t_h, X, q10, q25, qmed, q75, q90, qmean,
                                  samp_idx, cond_label, bucket):
            first_for_stat = True
            sp_colors = {}

            for si, sp in enumerate(species):
                for ridx in samp_idx:
                    ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha, label='_nolegend_')

                if show_p80:
                    band = ax.fill_between(
                        t_h, q10[:, si], q90[:, si],
                        alpha=band_alpha_p80, linewidth=0
                    )
                    if first_for_stat:
                        bucket[('P80', None)] = band
                if show_iqr:
                    band = ax.fill_between(
                        t_h, q25[:, si], q75[:, si],
                        alpha=band_alpha_iqr, linewidth=0
                    )
                    if first_for_stat:
                        bucket[('IQR', None)] = band

                if show_mean:
                    line = ax.plot(t_h, qmean[:, si], lw=2)[0]
                    sp_colors[sp] = line.get_color()
                    bucket[('mean', sp)] = line
                if show_median:
                    line = ax.plot(t_h, qmed[:, si], lw=1, linestyle='--')[0]
                    bucket[('median', sp)] = line

                first_for_stat = False

            if cond_label == 'Starved':
                ax.set_title(f"Starvation AXP Dynamics for N = {X.shape[0]} simulations", fontsize=15)
            else:
                ax.set_title(f"Basal State AXP Dynamics for N = {X.shape[0]} simulations", fontsize=15)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            ax.set_xlabel('time [h]', fontsize=12)
            ax.grid(alpha=0.2, linestyle=':')

            # left species labels
            if curve_labels and show_mean:
                y0s = [qmean[0, si] for si in range(len(species))]
                y_min, y_max = ax.get_ylim()
                y_range = max(1e-12, (y_max - y_min))
                y_labs = _spread(y0s, curve_label_min_gap_frac * y_range)

                x_min, x_max = ax.get_xlim()
                ax.set_xlim(0, x_max)

                for sp, y_lab, y_true in zip(species, y_labs, y0s):
                    col = sp_colors.get(sp, 'black')
                    ax.text(curve_label_xpad, y_lab, sp,
                            color=col, va='center', ha='right',
                            fontsize=12, weight='bold')
                    if curve_label_connectors:
                        ax.plot([curve_label_xpad + 0.2, 0],
                                [y_lab, y_true],
                                lw=2.0, alpha=0.6, color=col,
                                label='_nolegend_',
                                clip_on=False)

        _plot_condition_panel(axes[0], t_h_a, Xa, qa10, qa25, qamed, qa75, qa90, qamean,
                              samp_a, label_a, legend_data[label_a])
        _plot_condition_panel(axes[1], t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90, qbmean,
                              samp_b, label_b, legend_data[label_b])

        axes[0].set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)

        # --- NEW: force y-ticks/labels on the right subplot as well -----
        axes[1].tick_params(labelleft=True)
        # ----------------------------------------------------------------

        if show_axp and set(species) == {'ATP','ADP','AMP'}:
            line_a = axes[0].plot(t_h_a, qamean.sum(axis=1), lw=1.2, linestyle=':')[0]
            line_b = axes[1].plot(t_h_b, qbmean.sum(axis=1), lw=1.2, linestyle=':')[0]
            legend_data[label_a][('AXP_mean', None)] = line_a
            legend_data[label_b][('AXP_mean', None)] = line_b

    # -------- legend placement (same as your last version) -------------
    if layout_mode == "condition" and legend_pos in {"below", "above"}:
        cols = []
        if show_iqr:
            cols.append(('IQR', None))
        if show_p80:
            cols.append(('P80', None))
        for sp in species:
            if show_mean:
                cols.append(('mean', sp))
            if show_median:
                cols.append(('median', sp))
        if show_axp and set(species) == {'ATP','ADP','AMP'}:
            cols.append(('AXP_mean', None))

        def desc_str(key):
            kind, sp = key
            if kind == 'IQR':
                return 'IQR'
            if kind == 'P80':
                return '10–90%'
            if kind == 'mean' and sp is not None:
                return f'{sp} mean'
            if kind == 'median' and sp is not None:
                return f'{sp} median'
            if kind == 'AXP_mean':
                return 'AXP mean'
            return str(key)

        handles, labels = [], []
        for cond_label in (label_a, label_b):
            bucket = legend_data[cond_label]
            for key in cols:
                h = bucket.get(key)
                if h is not None:
                    handles.append(h)
                    labels.append(f"{cond_label} {desc_str(key)}")

        ncol = len(cols) if cols else 1
    else:
        if layout_mode == "condition":
            h0, l0 = axes[0].get_legend_handles_labels()
            h1, l1 = axes[1].get_legend_handles_labels()
            handles, labels = list(h0), list(l0)
            for h, lab in zip(h1, l1):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
        else:
            handles, labels = axes[0].get_legend_handles_labels()
        ncol = 6

    if legend_pos == "below":
        fig.legend(handles, labels, frameon=False, loc='lower center',
                   bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=12)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    elif legend_pos == "above":
        fig.legend(handles, labels, frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        axes[-1].legend(handles, labels, frameon=False, loc='upper right')
        fig.tight_layout()

    fig.patch.set_facecolor('white')
    #fig.subplots_adjust(left=0.22)
    return fig


def plot_compare_summary_ampk(
    df_dict_a: Dict[str, pd.DataFrame],
    df_dict_b: Dict[str, pd.DataFrame],
    label_a: str = 'cond A',
    label_b: str = 'cond B',
    species=('ATP','ADP','AMP'),
    dt_minutes=30,
    total_hours=84,
    # ---- layout switch ----
    layout_mode: str = "species",   # "species" | "condition"
    # ---- stat controls ----
    show_iqr: bool = True,
    show_p80: bool = False,
    show_mean: bool = True,
    show_median: bool = False,
    sample_trajs: int = 0,          # 0 -> no thin trajectories
    traj_alpha: float = 0.10,
    band_alpha_iqr: float = 0.20,
    band_alpha_p80: float = 0.10,
    legend_pos: str = "below",      # "below" | "above" | "inside"
    # -----------------------
    show_axp: bool = True,          # only used if plotting the AXP trio
    figsize=(12, 8),
    curve_labels: bool = False,
    curve_label_min_gap_frac: float = 0.02,
    curve_label_xpad: float = -3.0,
    curve_label_connectors: bool = True,
    # ---- separate figures ----
    multi_fig: bool = False,        # one fig per *species*
    separate_conditions: bool = False,  # only used if layout_mode="condition"
    user_title='',
    curve_label_xfrac=1,
    curve_label_y_offset_frac=0.02
):
    """
    Compare conditions A vs B for the given species.

    layout_mode = "species":
        One subplot per species. Each subplot shows both conditions.

    layout_mode = "condition":
        Two panels (A, B). Each panel shows all species for that condition.

    multi_fig = True:
        Instead of one figure for all species, return a list of figures, one
        per species (using the chosen layout_mode).

    separate_conditions = True (with layout_mode="condition"):
        Instead of a single figure with two subplots, return two separate
        figures: (fig_A, fig_B).
    """
    if layout_mode not in {"species", "condition"}:
        raise ValueError('layout_mode must be "species" or "condition".')

    species = _normalize_species(species)

    # --- one fig per species if requested -------------------
    if multi_fig and len(species) > 1:
        figs = []
        for sp in species:
            figs.append(
                plot_compare_summary(
                    df_dict_a=df_dict_a,
                    df_dict_b=df_dict_b,
                    label_a=label_a,
                    label_b=label_b,
                    species=(sp,),
                    dt_minutes=dt_minutes,
                    total_hours=total_hours,
                    layout_mode=layout_mode,
                    show_iqr=show_iqr,
                    show_p80=show_p80,
                    show_mean=show_mean,
                    show_median=show_median,
                    sample_trajs=sample_trajs,
                    traj_alpha=traj_alpha,
                    band_alpha_iqr=band_alpha_iqr,
                    band_alpha_p80=band_alpha_p80,
                    legend_pos=legend_pos,
                    show_axp=show_axp,
                    figsize=figsize,
                    curve_labels=curve_labels,
                    curve_label_min_gap_frac=curve_label_min_gap_frac,
                    curve_label_xpad=curve_label_xpad,
                    curve_label_connectors=curve_label_connectors,
                    multi_fig=False,
                    separate_conditions=separate_conditions,
                    user_title=user_title
                )
            )
        return figs
    # --------------------------------------------------------

    n_expected = int(total_hours*60/dt_minutes + 1)

    idx_a, Xa = _stack_runs(df_dict_a, species, n_expected)
    idx_b, Xb = _stack_runs(df_dict_b, species, n_expected)
    t_h_a = (idx_a.values - 1) * dt_minutes / 60.0
    t_h_b = (idx_b.values - 1) * dt_minutes / 60.0

    qa10, qa25, qamed, qa75, qa90, qamean = _summary(Xa)
    qb10, qb25, qbmed, qb75, qb90, qbmean = _summary(Xb)

    rng = np.random.default_rng(0)
    samp_a = rng.choice(Xa.shape[0], size=min(sample_trajs, Xa.shape[0]),
                        replace=False) if sample_trajs else []
    samp_b = rng.choice(Xb.shape[0], size=min(sample_trajs, Xb.shape[0]),
                        replace=False) if sample_trajs else []

    legend_data = {label_a: {}, label_b: {}}

    # ---------------- LAYOUT 1: one subplot per species ----------------
    if layout_mode == "species":
        fig, axes = plt.subplots(1, len(species), figsize=figsize, sharex=True)
        if len(species) == 1:
            axes = [axes]

        for si, sp in enumerate(species):
            ax = axes[si]

            for ridx in samp_a:
                ax.plot(t_h_a, Xa[ridx, :, si], lw=0.6, alpha=traj_alpha)
            for ridx in samp_b:
                ax.plot(t_h_b, Xb[ridx, :, si], lw=0.6, alpha=traj_alpha)

            if show_p80:
                ax.fill_between(t_h_a, qa10[:, si], qa90[:, si],
                                alpha=band_alpha_p80, linewidth=0,
                                label=f'{label_a} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_a, qa25[:, si], qa75[:, si],
                                alpha=band_alpha_iqr, linewidth=0,
                                label=f'{label_a} IQR')

            if show_p80:
                ax.fill_between(t_h_b, qb10[:, si], qb90[:, si],
                                alpha=band_alpha_p80, linewidth=0,
                                label=f'{label_b} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_b, qb25[:, si], qb75[:, si],
                                alpha=band_alpha_iqr, linewidth=0,
                                label=f'{label_b} IQR')

            if curve_labels:
                y0_a = qamean[0, si]
                y0_b = qbmean[0, si]
                ax.text(-2.5, y0_a, f"{label_a}", color='tab:blue',
                        va='center', ha='right', fontsize=12, weight='bold')
                ax.text(-2.5, y0_b, f"{label_b}", color='tab:orange',
                        va='center', ha='right', fontsize=12, weight='bold')

            if show_mean:
                ax.plot(t_h_a, qamean[:, si], lw=2, label=f'{label_a} mean')
                ax.plot(t_h_b, qbmean[:, si], lw=2, label=f'{label_b} mean')
            if show_median:
                ax.plot(t_h_a, qamed[:, si], lw=1, linestyle='--',
                        label=f'{label_a} median')
                ax.plot(t_h_b, qbmed[:, si], lw=1, linestyle='--',
                        label=f'{label_b} median')

            ax.set_title(
                f"Time Profile of {sp} Dynamics for N = {Xa.shape[0]} Simulations",
                fontsize=20
            )
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            if si == 0:
                ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
            ax.set_xlabel('time [h]', fontsize=12)
            ax.grid(alpha=0.2, linestyle=':')

        # legend for layout_mode="species"
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 6

        if legend_pos == "below":
            fig.legend(handles, labels, frameon=False, loc='lower center',
                       bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        elif legend_pos == "above":
            fig.legend(handles, labels, frameon=False, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
        elif legend_pos == 'none':
            pass
        else:
            axes[-1].legend(handles, labels, frameon=False, loc='upper right')
            fig.tight_layout()

        fig.patch.set_facecolor('white')
        return fig

    # ---------------- LAYOUT 2: condition layout ----------------
    # option A: separate figures per condition
    if separate_conditions:
        figs = []

        def _single_condition_fig(t_h, X, q10, q25, qmed, q75, q90, qmean,
                                  samp_idx, cond_label):
            fig_c, ax = plt.subplots(1, 1, figsize=figsize)

            sp_colors = {}
            for si, sp in enumerate(species):
                for ridx in samp_idx:
                    ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                            label='_nolegend_')

                if show_p80:
                    ax.fill_between(t_h, q10[:, si], q90[:, si],
                                    alpha=band_alpha_p80, linewidth=0,
                                    label='_nolegend_')
                if show_iqr:
                    ax.fill_between(t_h, q25[:, si], q75[:, si],
                                    alpha=band_alpha_iqr, linewidth=0,
                                    label='_nolegend_')

                if show_mean:
                    line = ax.plot(t_h, qmean[:, si], lw=2,
                                   label=f"{sp} mean")[0]
                    sp_colors[sp] = line.get_color()
                if show_median:
                    ax.plot(t_h, qmed[:, si], lw=1, linestyle='--',
                            label=f"{sp} median")

            if show_axp and set(species) == {'ATP','ADP','AMP'}: #####################################x
                ax.plot(t_h, qmean.sum(axis=1), lw=2.2, linestyle=':',
                        label='AXP mean')

            title = ("Starvation" if cond_label == 'Starved' else "Basal State") ############################x
            ax.set_title(f"{user_title} in {title}  for N = {X.shape[0]} simulations",
                         fontsize=20)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_xlabel('time [h]', fontsize=18)
            ax.set_ylabel(r'concentration [mM]', fontsize=20)
            ax.grid(alpha=0.2, linestyle=':')

            # left-side species labels if requested
                        # curve labels with connectors at arbitrary x-position
            if curve_labels and show_mean:
                # --- choose x-position along the curve ---
                x0, x1 = t_h[0], t_h[-1]
                if 0.0 <= curve_label_xfrac <= 1.0:
                    # interpret as fraction of [x0, x1]
                    x_target = x0 + curve_label_xfrac * (x1 - x0)
                else:
                    # interpret as absolute time in hours
                    x_target = curve_label_xfrac
                x_target = np.clip(x_target, x0, x1)

                # --- y-values of each curve at that x (via interpolation) ---
                y_at_x = [
                    np.interp(x_target, t_h, qmean[:, si])
                    for si in range(len(species))
                ]

                # --- spread vertically to avoid overlaps ---
                y_min, y_max = ax.get_ylim()
                y_range = max(1e-12, (y_max - y_min))
                y_spread = _spread(y_at_x, curve_label_min_gap_frac * y_range)

                # optional extra offset above/below the curve
                y_labels = [
                    y_s + curve_label_y_offset_frac * y_range
                    for y_s in y_spread
                ]

                # x-position of labels: a bit to the right of x_target
                x_label = x_target + curve_label_xpad

                # extend x-limits if labels would be outside the axes
                if x_label > x1:
                    ax.set_xlim(x0, x_label + 0.5)

                for sp, y_lab, y_true in zip(species, y_labels, y_at_x):
                    col = sp_colors.get(sp, 'black')
                    ax.text(
                        x_label, y_lab, sp,
                        color=col, va='center', ha='left',
                        fontsize=12, weight='bold'
                    )
                    if curve_label_connectors:
                        ax.plot(
                            [x_target, x_label - 0.2],
                            [y_true,  y_lab],
                            lw=2.0, alpha=0.6, color=col,
                            label='_nolegend_',
                            clip_on=False
                        )


            handles_c, labels_c = ax.get_legend_handles_labels()
            if legend_pos == "below":
                fig_c.legend(handles_c, labels_c, frameon=False, loc='lower center',
                             bbox_to_anchor=(0.5, -0.01), ncol=len(labels_c),
                             fontsize=12)
                fig_c.tight_layout(rect=[0, 0.06, 1, 1])
            elif legend_pos == "above":
                fig_c.legend(handles_c, labels_c, frameon=False, loc='upper center',
                             bbox_to_anchor=(0.5, 1.15), ncol=len(labels_c),
                             fontsize=12)
                fig_c.tight_layout(rect=[0, 0, 1, 0.94])
            elif legend_pos == 'none':
                pass
            else:
                ax.legend(frameon=False, loc='upper right')
                fig_c.tight_layout()

            fig_c.patch.set_facecolor('white')
            return fig_c

        figs.append(_single_condition_fig(
            t_h_a, Xa, qa10, qa25, qamed, qa75, qa90, qamean,
            samp_a, label_a
        ))
        figs.append(_single_condition_fig(
            t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90, qbmean,
            samp_b, label_b
        ))
        return figs

    # option B: original behaviour – one figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    def _plot_condition_panel(ax, t_h, X, q10, q25, qmed, q75, q90, qmean,
                              samp_idx, cond_label, bucket):
        first_for_stat = True
        sp_colors = {}

        for si, sp in enumerate(species):
            for ridx in samp_idx:
                ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                        label='_nolegend_')

            if show_p80:
                band = ax.fill_between(
                    t_h, q10[:, si], q90[:, si],
                    alpha=band_alpha_p80, linewidth=0
                )
                if first_for_stat:
                    bucket[('P80', None)] = band
            if show_iqr:
                band = ax.fill_between(
                    t_h, q25[:, si], q75[:, si],
                    alpha=band_alpha_iqr, linewidth=0
                )
                if first_for_stat:
                    bucket[('IQR', None)] = band

            if show_mean:
                line = ax.plot(t_h, qmean[:, si], lw=2)[0]
                sp_colors[sp] = line.get_color()
                bucket[('mean', sp)] = line
            if show_median:
                line = ax.plot(t_h, qmed[:, si], lw=1, linestyle='--')[0]
                bucket[('median', sp)] = line

            first_for_stat = False

        if cond_label == 'Starved':
            ax.set_title(
                f"AXP Dynamics during Starvation  for N = {X.shape[0]} simulations",
                fontsize=15
            )
        else:
            ax.set_title(
                f" AXP Dynamics in Basal State for N = {X.shape[0]} simulations",
                fontsize=15
            )
        ax.set_xlim(0, total_hours)
        ax.set_xticks(np.arange(0, total_hours+0.1, 6))
        ax.set_xlabel('time [h]', fontsize=12)
        ax.grid(alpha=0.2, linestyle=':')

        if curve_labels and show_mean:
            y0s = [qmean[0, si] for si in range(len(species))]
            y_min, y_max = ax.get_ylim()
            y_range = max(1e-12, (y_max - y_min))
            y_labs = _spread(y0s, curve_label_min_gap_frac * y_range)

            for sp, y_lab, y_true in zip(species, y_labs, y0s):
                col = sp_colors.get(sp, 'black')
                ax.text(curve_label_xpad, y_lab, sp,
                        color=col, va='center', ha='right',
                        fontsize=12, weight='bold')
                if curve_label_connectors:
                    ax.plot([curve_label_xpad + 0.2, 0],
                            [y_lab, y_true],
                            lw=2.0, alpha=0.6, color=col,
                            label='_nolegend_',
                            clip_on=False)

    _plot_condition_panel(axes[0], t_h_a, Xa, qa10, qa25, qamed, qa75, qa90,
                          qamean, samp_a, label_a, legend_data[label_a])
    _plot_condition_panel(axes[1], t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90,
                          qbmean, samp_b, label_b, legend_data[label_b])

    axes[0].set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
    axes[1].tick_params(labelleft=True)

    if show_axp and set(species) == {'ATP','ADP','AMP'}:
        line_a = axes[0].plot(t_h_a, qamean.sum(axis=1),
                              lw=1.2, linestyle=':')[0]
        line_b = axes[1].plot(t_h_b, qbmean.sum(axis=1),
                              lw=1.2, linestyle=':')[0]
        legend_data[label_a][('AXP_mean', None)] = line_a
        legend_data[label_b][('AXP_mean', None)] = line_b

    # ---- shared legend for this layout ----
    if legend_pos in {"below", "above"}:
        cols = []
        if show_iqr:
            cols.append(('IQR', None))
        if show_p80:
            cols.append(('P80', None))
        for sp in species:
            if show_mean:
                cols.append(('mean', sp))
            if show_median:
                cols.append(('median', sp))
        if show_axp and set(species) == {'ATP','ADP','AMP'}:
            cols.append(('AXP_mean', None))

        def desc_str(key):
            kind, sp = key
            if kind == 'IQR':
                return 'IQR'
            if kind == 'P80':
                return '10–90%'
            if kind == 'mean' and sp is not None:
                return f'{sp} mean'
            if kind == 'median' and sp is not None:
                return f'{sp} median'
            if kind == 'AXP_mean':
                return 'AXP mean'
            return str(key)

        handles, labels = [], []
        for cond_label in (label_a, label_b):
            bucket = legend_data[cond_label]
            for key in cols:
                h = bucket.get(key)
                if h is not None:
                    handles.append(h)
                    labels.append(f"{cond_label} {desc_str(key)}")

        ncol = len(cols) if cols else 1

        if legend_pos == "below":
            fig.legend(handles, labels, frameon=False, loc='lower center',
                       bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        else:  # "above"
            fig.legend(handles, labels, frameon=False, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        axes[-1].legend(frameon=False, loc='upper right')
        fig.tight_layout()

    fig.patch.set_facecolor('white')
    return fig


def add_ampk_totals(df_dict,
                    ampk_name='AMPK',
                    ampka_name='AMPKA'):
    """
    Return a new dict of DataFrames where each df has two extra columns:
    AMPK_tot  = AMPK  + AMPK_ATP  + AMPK_ADP  + AMPK_AMP
    AMPKA_tot = AMPKA + AMPKA_ATP + AMPKA_ADP + AMPKA_AMP
    """
    out = {}
    for key, df in df_dict.items():
        tmp = df.copy()

        tmp[ampk_name] = (
            tmp['AMPK'] +
            tmp['AMPK_ATP'] +
            tmp['AMPK_ADP'] +
            tmp['AMPK_AMP']
        )

        tmp[ampka_name] = (
            tmp['AMPKA'] +
            tmp['AMPKA_ATP'] +
            tmp['AMPKA_ADP'] +
            tmp['AMPKA_AMP']
        )

        out[key] = tmp
    return out


def plot_compare_summary_atp(
    df_dict_a: Dict[str, pd.DataFrame],
    df_dict_b: Dict[str, pd.DataFrame],
    label_a: str = 'cond A',
    label_b: str = 'cond B',
    species=('ATP','ADP','AMP'),
    dt_minutes=30,
    total_hours=84,
    # ---- layout switch ----
    layout_mode: str = "species",   # "species" | "condition"
    # ---- stat controls ----
    show_iqr: bool = True,
    show_p80: bool = False,
    show_mean: bool = True,
    show_median: bool = False,
    sample_trajs: int = 0,          # 0 -> no thin trajectories
    traj_alpha: float = 0.10,
    band_alpha_iqr: float = 0.20,
    band_alpha_p80: float = 0.10,
    legend_pos: str = "below",      # "below" | "above" | "inside"
    # -----------------------
    show_axp: bool = True,          # only used if plotting the AXP trio
    figsize=(12, 8),
    curve_labels: bool = False,
    curve_label_min_gap_frac: float = 0.02,
    curve_label_xpad: float = -3.0,
    curve_label_connectors: bool = True,
    # ---- separate figures ----
    multi_fig: bool = False,        # one fig per *species*
    separate_conditions: bool = False,  # only used if layout_mode="condition"
    user_title='',
    title_font_size=15,
    x_title_font_size=10
):
    """
    Compare conditions A vs B for the given species.

    layout_mode = "species":
        One subplot per species. Each subplot shows both conditions.

    layout_mode = "condition":
        Two panels (A, B). Each panel shows all species for that condition.

    multi_fig = True:
        Instead of one figure for all species, return a list of figures, one
        per species (using the chosen layout_mode).

    separate_conditions = True (with layout_mode="condition"):
        Instead of a single figure with two subplots, return two separate
        figures: (fig_A, fig_B).
    """
    if layout_mode not in {"species", "condition"}:
        raise ValueError('layout_mode must be "species" or "condition".')

    species = _normalize_species(species)

    # --- one fig per species if requested -------------------
    if multi_fig and len(species) > 1:
        figs = []
        for sp in species:
            figs.append(
                plot_compare_summary(
                    df_dict_a=df_dict_a,
                    df_dict_b=df_dict_b,
                    label_a=label_a,
                    label_b=label_b,
                    species=(sp,),
                    dt_minutes=dt_minutes,
                    total_hours=total_hours,
                    layout_mode=layout_mode,
                    show_iqr=show_iqr,
                    show_p80=show_p80,
                    show_mean=show_mean,
                    show_median=show_median,
                    sample_trajs=sample_trajs,
                    traj_alpha=traj_alpha,
                    band_alpha_iqr=band_alpha_iqr,
                    band_alpha_p80=band_alpha_p80,
                    legend_pos=legend_pos,
                    show_axp=show_axp,
                    figsize=figsize,
                    curve_labels=curve_labels,
                    curve_label_min_gap_frac=curve_label_min_gap_frac,
                    curve_label_xpad=curve_label_xpad,
                    curve_label_connectors=curve_label_connectors,
                    multi_fig=False,
                    separate_conditions=separate_conditions,
                    user_title=user_title,
                    title_font_size=title_font_size,
                    x_title_font_size=x_title_font_size
                )
            )
        return figs
    # --------------------------------------------------------

    n_expected = int(total_hours*60/dt_minutes + 1)

    idx_a, Xa = _stack_runs(df_dict_a, species, n_expected)
    idx_b, Xb = _stack_runs(df_dict_b, species, n_expected)
    t_h_a = (idx_a.values - 1) * dt_minutes / 60.0
    t_h_b = (idx_b.values - 1) * dt_minutes / 60.0

    qa10, qa25, qamed, qa75, qa90, qamean = _summary(Xa)
    qb10, qb25, qbmed, qb75, qb90, qbmean = _summary(Xb)

    rng = np.random.default_rng(0)
    samp_a = rng.choice(Xa.shape[0], size=min(sample_trajs, Xa.shape[0]),
                        replace=False) if sample_trajs else []
    samp_b = rng.choice(Xb.shape[0], size=min(sample_trajs, Xb.shape[0]),
                        replace=False) if sample_trajs else []

    legend_data = {label_a: {}, label_b: {}}

    # ---------------- LAYOUT 1: one subplot per species ----------------
    if layout_mode == "species":
        fig, axes = plt.subplots(1, len(species), figsize=figsize, sharex=True)
        if len(species) == 1:
            axes = [axes]

        for si, sp in enumerate(species):
            ax = axes[si]

            for ridx in samp_a:
                ax.plot(t_h_a, Xa[ridx, :, si], lw=0.6, alpha=traj_alpha)
            for ridx in samp_b:
                ax.plot(t_h_b, Xb[ridx, :, si], lw=0.6, alpha=traj_alpha)

            if show_p80:
                ax.fill_between(t_h_a, qa10[:, si], qa90[:, si],
                                alpha=band_alpha_p80, linewidth=0,
                                label=f'{label_a} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_a, qa25[:, si], qa75[:, si],
                                alpha=band_alpha_iqr, linewidth=0,
                                label=f'{label_a} IQR')

            if show_p80:
                ax.fill_between(t_h_b, qb10[:, si], qb90[:, si],
                                alpha=band_alpha_p80, linewidth=0,
                                label=f'{label_b} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_b, qb25[:, si], qb75[:, si],
                                alpha=band_alpha_iqr, linewidth=0,
                                label=f'{label_b} IQR')

            if curve_labels:
                y0_a = qamean[0, si]
                y0_b = qbmean[0, si]
                ax.text(-2.5, y0_a, f"{label_a}", color='tab:blue',
                        va='center', ha='right', fontsize=12, weight='bold')
                ax.text(-2.5, y0_b, f"{label_b}", color='tab:orange',
                        va='center', ha='right', fontsize=12, weight='bold')

            if show_mean:
                ax.plot(t_h_a, qamean[:, si], lw=2, label=f'{label_a} mean')
                ax.plot(t_h_b, qbmean[:, si], lw=2, label=f'{label_b} mean')
            if show_median:
                ax.plot(t_h_a, qamed[:, si], lw=1, linestyle='--',
                        label=f'{label_a} median')
                ax.plot(t_h_b, qbmed[:, si], lw=1, linestyle='--',
                        label=f'{label_b} median')

            ax.set_title(
                f"Time Profile of {sp} Dynamics for N = {Xa.shape[0]} Simulations",
                fontsize=20
            )
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            if si == 0:
                ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
            ax.set_xlabel('time [h]', fontsize=x_title_font_size)
            ax.grid(alpha=0.2, linestyle=':')

        # legend for layout_mode="species"
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 6

        if legend_pos == "below":
            fig.legend(handles, labels, frameon=False, loc='lower center',
                       bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        elif legend_pos == "above":
            fig.legend(handles, labels, frameon=False, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
        else:
            axes[-1].legend(handles, labels, frameon=False, loc='upper right')
            fig.tight_layout()

        fig.patch.set_facecolor('white')
        return fig

    # ---------------- LAYOUT 2: condition layout ----------------
    # option A: separate figures per condition
    if separate_conditions:
        figs = []

        def _single_condition_fig(t_h, X, q10, q25, qmed, q75, q90, qmean,
                                  samp_idx, cond_label):
            fig_c, ax = plt.subplots(1, 1, figsize=figsize)

            sp_colors = {}
            for si, sp in enumerate(species):
                for ridx in samp_idx:
                    ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                            label='_nolegend_')

                if show_p80:
                    ax.fill_between(t_h, q10[:, si], q90[:, si],
                                    alpha=band_alpha_p80, linewidth=0,
                                    label='_nolegend_')
                if show_iqr:
                    ax.fill_between(t_h, q25[:, si], q75[:, si],
                                    alpha=band_alpha_iqr, linewidth=0,
                                    label='_nolegend_')

                if show_mean:
                    line = ax.plot(t_h, qmean[:, si], lw=2,
                                   label=f"{sp} mean")[0]
                    sp_colors[sp] = line.get_color()
                if show_median:
                    ax.plot(t_h, qmed[:, si], lw=1, linestyle='--',
                            label=f"{sp} median")

            if show_axp and set(species) == {'ATP','ADP','AMP'}: #####################################x
                ax.plot(t_h, qmean.sum(axis=1), lw=2.2, linestyle=':',
                        label='AXP mean')

            title = ("Starvation" if cond_label == 'Starved' else "Basal State") ############################x
            ax.set_title(f"{user_title} in {title}  for N = {X.shape[0]} simulations",
                         fontsize=title_font_size)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            ax.set_xlabel('time [h]', fontsize=16)
            ax.set_ylabel(r'concentration [mM]', fontsize=18)
            ax.grid(alpha=0.2, linestyle=':')
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)

            # left-side species labels if requested
            if curve_labels and show_mean:
                y0s = [qmean[0, si] for si in range(len(species))]
                y_min, y_max = ax.get_ylim()
                y_range = max(1e-12, (y_max - y_min))
                y_labs = _spread(y0s, curve_label_min_gap_frac * y_range)

                for sp, y_lab, y_true in zip(species, y_labs, y0s):
                    col = sp_colors.get(sp, 'black')
                    ax.text(curve_label_xpad, y_lab, sp,
                            color=col, va='center', ha='right',
                            fontsize=12, weight='bold')
                    if curve_label_connectors:
                        ax.plot([curve_label_xpad + 0.2, 0],
                                [y_lab, y_true],
                                lw=2.0, alpha=0.6, color=col,
                                label='_nolegend_',
                                clip_on=False)

            handles_c, labels_c = ax.get_legend_handles_labels()
            if legend_pos == "below":
                fig_c.legend(handles_c, labels_c, frameon=False, loc='lower center',
                             bbox_to_anchor=(0.5, -0.01), ncol=len(labels_c),
                             fontsize=12)
                fig_c.tight_layout(rect=[0, 0.06, 1, 1])
            elif legend_pos == "above":
                fig_c.legend(handles_c, labels_c, frameon=False, loc='upper center',
                             bbox_to_anchor=(0.5, 1.15), ncol=len(labels_c),
                             fontsize=12)
                fig_c.tight_layout(rect=[0, 0, 1, 0.94])
            else:
                ax.legend(frameon=False, loc='upper right')
                fig_c.tight_layout()

            fig_c.patch.set_facecolor('white')
            return fig_c

        figs.append(_single_condition_fig(
            t_h_a, Xa, qa10, qa25, qamed, qa75, qa90, qamean,
            samp_a, label_a
        ))
        figs.append(_single_condition_fig(
            t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90, qbmean,
            samp_b, label_b
        ))
        return figs

    # option B: original behaviour – one figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    def _plot_condition_panel(ax, t_h, X, q10, q25, qmed, q75, q90, qmean,
                              samp_idx, cond_label, bucket):
        first_for_stat = True
        sp_colors = {}

        for si, sp in enumerate(species):
            for ridx in samp_idx:
                ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                        label='_nolegend_')

            if show_p80:
                band = ax.fill_between(
                    t_h, q10[:, si], q90[:, si],
                    alpha=band_alpha_p80, linewidth=0
                )
                if first_for_stat:
                    bucket[('P80', None)] = band
            if show_iqr:
                band = ax.fill_between(
                    t_h, q25[:, si], q75[:, si],
                    alpha=band_alpha_iqr, linewidth=0
                )
                if first_for_stat:
                    bucket[('IQR', None)] = band

            if show_mean:
                line = ax.plot(t_h, qmean[:, si], lw=2)[0]
                sp_colors[sp] = line.get_color()
                bucket[('mean', sp)] = line
            if show_median:
                line = ax.plot(t_h, qmed[:, si], lw=1, linestyle='--')[0]
                bucket[('median', sp)] = line

            first_for_stat = False

        if cond_label == 'Starved':
            ax.set_title(
                f"AXP Dynamics during Starvation  for N = {X.shape[0]} simulations",
                fontsize=title_font_size
            )
        else:
            ax.set_title(
                f" AXP Dynamics in Basal State for N = {X.shape[0]} simulations",
                fontsize=title_font_size
            )
        ax.set_xlim(0, total_hours)
        ax.set_xticks(np.arange(0, total_hours+0.1, 6))
        ax.set_xlabel('time [h]', fontsize=x_title_font_size)
        ax.grid(alpha=0.2, linestyle=':')

        if curve_labels and show_mean:
            y0s = [qmean[0, si] for si in range(len(species))]
            y_min, y_max = ax.get_ylim()
            y_range = max(1e-12, (y_max - y_min))
            y_labs = _spread(y0s, curve_label_min_gap_frac * y_range)

            for sp, y_lab, y_true in zip(species, y_labs, y0s):
                col = sp_colors.get(sp, 'black')
                ax.text(curve_label_xpad, y_lab, sp,
                        color=col, va='center', ha='right',
                        fontsize=12, weight='bold')
                if curve_label_connectors:
                    ax.plot([curve_label_xpad + 0.2, 0],
                            [y_lab, y_true],
                            lw=2.0, alpha=0.6, color=col,
                            label='_nolegend_',
                            clip_on=False)

    _plot_condition_panel(axes[0], t_h_a, Xa, qa10, qa25, qamed, qa75, qa90,
                          qamean, samp_a, label_a, legend_data[label_a])
    _plot_condition_panel(axes[1], t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90,
                          qbmean, samp_b, label_b, legend_data[label_b])

    axes[0].set_ylabel(r'concentration [mM]', fontsize=15)
    axes[1].tick_params(labelleft=True)
    axes[0].tick_params(axis='x', labelsize=18)
    axes[0].tick_params(axis='y', labelsize=18)
    axes[1].tick_params(axis='x', labelsize=18)
    axes[1].tick_params(axis='y', labelsize=18)

    if show_axp and set(species) == {'ATP','ADP','AMP'}:
        line_a = axes[0].plot(t_h_a, qamean.sum(axis=1),
                              lw=1.2, linestyle=':')[0]
        line_b = axes[1].plot(t_h_b, qbmean.sum(axis=1),
                              lw=1.2, linestyle=':')[0]
        legend_data[label_a][('AXP_mean', None)] = line_a
        legend_data[label_b][('AXP_mean', None)] = line_b

    # ---- shared legend for this layout ----
    if legend_pos in {"below", "above"}:
        cols = []
        if show_iqr:
            cols.append(('IQR', None))
        if show_p80:
            cols.append(('P80', None))
        for sp in species:
            if show_mean:
                cols.append(('mean', sp))
            if show_median:
                cols.append(('median', sp))
        if show_axp and set(species) == {'ATP','ADP','AMP'}:
            cols.append(('AXP_mean', None))

        def desc_str(key):
            kind, sp = key
            if kind == 'IQR':
                return 'IQR'
            if kind == 'P80':
                return '10–90%'
            if kind == 'mean' and sp is not None:
                return f'{sp} mean'
            if kind == 'median' and sp is not None:
                return f'{sp} median'
            if kind == 'AXP_mean':
                return 'AXP mean'
            return str(key)

        handles, labels = [], []
        for cond_label in (label_a, label_b):
            bucket = legend_data[cond_label]
            for key in cols:
                h = bucket.get(key)
                if h is not None:
                    handles.append(h)
                    labels.append(f"{cond_label} {desc_str(key)}")

        ncol = len(cols) if cols else 1

        if legend_pos == "below":
            fig.legend(handles, labels, frameon=False, loc='lower center',
                       bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        else:  # "above"
            fig.legend(handles, labels, frameon=False, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        axes[-1].legend(frameon=False, loc='upper right')
        fig.tight_layout()

    fig.patch.set_facecolor('white')
    return fig


# %%
def add_ampk_totals(df_dict,
                    ampk_name='AMPK_t',
                    ampka_name='AMPKA_t'):
    """
    Return a new dict of DataFrames where each df has two extra columns:
    AMPK_tot  = AMPK  + AMPK_ATP  + AMPK_ADP  + AMPK_AMP
    AMPKA_tot = AMPKA + AMPKA_ATP + AMPKA_ADP + AMPKA_AMP
    """
    out = {}
    for key, df in df_dict.items():
        tmp = df.copy()

        tmp[ampk_name] = (
            tmp['AMPK'] +
            tmp['AMPK_ATP'] +
            tmp['AMPK_ADP'] +
            tmp['AMPK_AMP']
        )

        tmp[ampka_name] = (
            tmp['AMPKA'] +
            tmp['AMPKA_ATP'] +
            tmp['AMPKA_ADP'] +
            tmp['AMPKA_AMP']
        )

        out[key] = tmp
    return out

stac_ampk   = add_ampk_totals(mechs['stac'].orig_time_sim_df)
starve_ampk = add_ampk_totals(mechs['starve'].orig_time_sim_df)

figs_mau = plot_compare_summary_ampk(
    df_dict_a=starve_ampk,
    df_dict_b=stac_ampk,
    label_a='Starved',
    label_b='Basal',
    species=('AMPKA_t', 'ULKA', 'MTORA'),
    layout_mode="condition",   # or "species", as you like
    show_iqr=False, show_p80=False,
    show_mean=True, show_median=False,
    sample_trajs=0,
    legend_pos="none",
    separate_conditions=True, 
    curve_labels=False,
    curve_label_xfrac=20,        # 1.0 = end of time window
    curve_label_xpad=1.5,         # 1.5 h to the right of that point
    curve_label_y_offset_frac=0.02,  # 2% of y-range above the curve
    figsize=(12, 8),
    user_title='AMPK-MTOR-ULK Dynamics'
)


# %%
def plot_compare_summary_ampk_conceptual(
    df_dict_a,
    df_dict_b,
    label_a: str = 'cond A',
    label_b: str = 'cond B',
    species=('ATP','ADP','AMP'),
    dt_minutes=30,
    total_hours=84,
    # ---- layout switch ----
    layout_mode: str = "species",   # "species" | "condition"
    # ---- stat controls ----
    show_iqr: bool = True,
    show_p80: bool = False,
    show_mean: bool = True,
    show_median: bool = False,
    sample_trajs: int = 0,          # 0 -> no thin trajectories
    traj_alpha: float = 0.10,
    band_alpha_iqr: float = 0.20,
    band_alpha_p80: float = 0.10,
    legend_pos: str = "below",      # "below" | "above" | "inside" | "none"
    show_axp: bool = True,          # only used if plotting the AXP trio
    figsize=(12, 8),
    display_dpi: int = 150,
    curve_labels: bool = False,
    curve_label_min_gap_frac: float = 0.02,
    curve_label_xpad: float = -3.0,
    curve_label_connectors: bool = True,
    legend_fontsize=12,
    # ---- separate figures ----
    multi_fig: bool = False,        # one fig per *species*
    separate_conditions: bool = False,  # only used if layout_mode="condition"
    user_title='',
    curve_label_xfrac=1,
    curve_label_y_offset_frac=0.02,
    # ---- experimental overlay ----
    exp_data_a: dict | None = None,     # {sp: {"t":..., "y":..., "yerr":...}, ...}
    exp_data_b: dict | None = None,
    exp_auto: bool = True,              # auto-generate exp points if exp_data_* is None
    exp_n_points: int = 9,
    exp_time_points_h=None,             # array-like or None
    exp_noise_frac: float = 0.1,       # closeness to mean (relative)
    exp_err_frac: float = 0.15,         # errorbar size (relative)
    exp_seed: int = 7,
    exp_markersize: float = 6,
    exp_capsize: float = 3,
    exp_zorder: int = 6,
):
    """
    Compare conditions A vs B for the given species, with optional experimental datapoints
    (error bars) overlaid using the same color as the simulated mean curve.

    Experimental data format (per condition):
      exp_data_a = {
        "AMPKA_t": {"t": np.array([...]), "y": np.array([...]), "yerr": np.array([...])},
        "ULKA":    {"t": ..., "y": ..., "yerr": ...},
        ...
      }

    If exp_auto=True and exp_data_a/b is None, synthetic experimental points are generated
    close to the simulated mean (useful for mock overlays).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---------------- helpers ----------------
    def _normalize_species(species_):
        if isinstance(species_, str):
            return (species_,)
        return tuple(species_)

    def _stack_runs(df_dict, species_, n_expected=None):
        species_ = _normalize_species(species_)
        max_idx = max(df.index.max() for df in df_dict.values())
        if n_expected is not None:
            max_idx = max(max_idx, n_expected)
        idx = pd.RangeIndex(1, max_idx + 1)

        mats = []
        for _, df in df_dict.items():
            tmp = (df.sort_index()
                     .reindex(idx)
                     .apply(pd.to_numeric, errors='coerce')
                     .ffill().bfill())
            if set(species_).issubset(tmp.columns):
                mats.append(tmp.loc[:, list(species_)].to_numpy())
        if not mats:
            raise ValueError(f"No runs contained the requested species: {species_}")
        return idx, np.stack(mats, axis=0)

    def _summary(X: np.ndarray):
        q10 = np.quantile(X, 0.10, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        med = np.quantile(X, 0.50, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        q90 = np.quantile(X, 0.90, axis=0)
        mean = X.mean(axis=0)
        return q10, q25, med, q75, q90, mean

    def _spread(y_values, min_gap):
        y = np.array(y_values, dtype=float)
        order = np.argsort(y)
        y_sorted = y[order]

        adj = np.empty_like(y_sorted)
        prev = -np.inf
        for i, val in enumerate(y_sorted):
            if val <= prev + min_gap:
                adj[i] = prev + min_gap
            else:
                adj[i] = val
            prev = adj[i]

        result = np.empty_like(adj)
        result[order] = adj
        return result

    def _auto_experiment_from_mean(
        t_h: np.ndarray,
        qmean: np.ndarray,
        species_: tuple,
        n_points: int = 8,
        time_points_h=None,
        noise_frac: float = 0.03,
        err_frac: float = 0.06,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        x0, x1 = float(t_h[0]), float(t_h[-1])

        if time_points_h is None:
            t_pts = np.linspace(x0 + 0.05*(x1-x0), x1 - 0.05*(x1-x0), n_points)
        else:
            t_pts = np.asarray(time_points_h, dtype=float)
            t_pts = np.clip(t_pts, x0, x1)

        exp = {}
        for si, sp in enumerate(species_):
            y_true = np.interp(t_pts, t_h, qmean[:, si])

            scale = np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            y_pts = y_true + rng.normal(0.0, noise_frac * scale)

            yerr = err_frac * np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            exp[sp] = {"t": t_pts, "y": y_pts, "yerr": yerr}
        return exp

    # ---------------- validate ----------------
    if layout_mode not in {"species", "condition"}:
        raise ValueError('layout_mode must be "species" or "condition".')

    species = _normalize_species(species)

    # --- one fig per species if requested -------------------
    if multi_fig and len(species) > 1:
        figs = []
        for sp in species:
            figs.append(
                plot_compare_summary_ampk(
                    df_dict_a=df_dict_a,
                    df_dict_b=df_dict_b,
                    label_a=label_a,
                    label_b=label_b,
                    species=(sp,),
                    dt_minutes=dt_minutes,
                    total_hours=total_hours,
                    layout_mode=layout_mode,
                    show_iqr=show_iqr,
                    show_p80=show_p80,
                    show_mean=show_mean,
                    show_median=show_median,
                    sample_trajs=sample_trajs,
                    traj_alpha=traj_alpha,
                    band_alpha_iqr=band_alpha_iqr,
                    band_alpha_p80=band_alpha_p80,
                    legend_pos=legend_pos,
                    show_axp=show_axp,
                    figsize=figsize,
                    display_dpi=display_dpi,
                    curve_labels=curve_labels,
                    curve_label_min_gap_frac=curve_label_min_gap_frac,
                    curve_label_xpad=curve_label_xpad,
                    curve_label_connectors=curve_label_connectors,
                    multi_fig=False,
                    separate_conditions=separate_conditions,
                    user_title=user_title,
                    curve_label_xfrac=curve_label_xfrac,
                    curve_label_y_offset_frac=curve_label_y_offset_frac,
                    # exp passthrough
                    exp_data_a=exp_data_a,
                    exp_data_b=exp_data_b,
                    exp_auto=exp_auto,
                    exp_n_points=exp_n_points,
                    exp_time_points_h=exp_time_points_h,
                    exp_noise_frac=exp_noise_frac,
                    exp_err_frac=exp_err_frac,
                    exp_seed=exp_seed,
                    exp_markersize=exp_markersize,
                    exp_capsize=exp_capsize,
                    exp_zorder=exp_zorder,
                )
            )
        return figs
    # --------------------------------------------------------

    n_expected = int(total_hours*60/dt_minutes + 1)

    idx_a, Xa = _stack_runs(df_dict_a, species, n_expected)
    idx_b, Xb = _stack_runs(df_dict_b, species, n_expected)
    t_h_a = (idx_a.values - 1) * dt_minutes / 60.0
    t_h_b = (idx_b.values - 1) * dt_minutes / 60.0

    #Xa = Xa * 1e+6
    #Xb = Xb * 1e+6

    qa10, qa25, qamed, qa75, qa90, qamean = _summary(Xa)
    qb10, qb25, qbmed, qb75, qb90, qbmean = _summary(Xb)

    # --- experimental data: user-supplied or auto-generated ---
    if exp_auto and exp_data_a is None:
        exp_data_a = _auto_experiment_from_mean(
            t_h=t_h_a,
            qmean=qamean,
            species_=species,
            n_points=exp_n_points,
            time_points_h=exp_time_points_h,
            noise_frac=exp_noise_frac,
            err_frac=exp_err_frac,
            seed=exp_seed,
        )
    if exp_auto and exp_data_b is None:
        exp_data_b = _auto_experiment_from_mean(
            t_h=t_h_b,
            qmean=qbmean,
            species_=species,
            n_points=exp_n_points,
            time_points_h=exp_time_points_h,
            noise_frac=exp_noise_frac,
            err_frac=exp_err_frac,
            seed=exp_seed + 1,
        )

    rng = np.random.default_rng(0)
    samp_a = rng.choice(Xa.shape[0], size=min(sample_trajs, Xa.shape[0]),
                        replace=False) if sample_trajs else []
    samp_b = rng.choice(Xb.shape[0], size=min(sample_trajs, Xb.shape[0]),
                        replace=False) if sample_trajs else []

    legend_data = {label_a: {}, label_b: {}}

    # ---------------- LAYOUT 1: one subplot per species ----------------
    if layout_mode == "species":
        fig, axes = plt.subplots(1, len(species), figsize=figsize, sharex=True)
        fig.set_dpi(display_dpi)
        if len(species) == 1:
            axes = [axes]

        for si, sp in enumerate(species):
            ax = axes[si]

            for ridx in samp_a:
                ax.plot(t_h_a, Xa[ridx, :, si], lw=0.6, alpha=traj_alpha)
            for ridx in samp_b:
                ax.plot(t_h_b, Xb[ridx, :, si], lw=0.6, alpha=traj_alpha)

            if show_p80:
                ax.fill_between(t_h_a, qa10[:, si], qa90[:, si],
                                alpha=band_alpha_p80, linewidth=0,
                                label=f'{label_a} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_a, qa25[:, si], qa75[:, si],
                                alpha=band_alpha_iqr, linewidth=0,
                                label=f'{label_a} IQR')

            if show_p80:
                ax.fill_between(t_h_b, qb10[:, si], qb90[:, si],
                                alpha=band_alpha_p80, linewidth=0,
                                label=f'{label_b} 10–90%')
            if show_iqr:
                ax.fill_between(t_h_b, qb25[:, si], qb75[:, si],
                                alpha=band_alpha_iqr, linewidth=0,
                                label=f'{label_b} IQR')

            sp_colors = {}
            if show_mean:
                line_a = ax.plot(t_h_a, qamean[:, si], lw=2, label=f'{label_a} mean')[0]
                line_b = ax.plot(t_h_b, qbmean[:, si], lw=2, label=f'{label_b} mean')[0]
                sp_colors[label_a] = line_a.get_color()
                sp_colors[label_b] = line_b.get_color()
            if show_median:
                ax.plot(t_h_a, qamed[:, si], lw=1, linestyle='--', label=f'{label_a} median')
                ax.plot(t_h_b, qbmed[:, si], lw=1, linestyle='--', label=f'{label_b} median')

            # experimental overlay (same color as each condition’s curve)
            if exp_data_a is not None and sp in exp_data_a and show_mean:
                ax.errorbar(
                    np.asarray(exp_data_a[sp]["t"]), np.asarray(exp_data_a[sp]["y"]),
                    yerr=np.asarray(exp_data_a[sp]["yerr"]),
                    fmt='o', linestyle='none',
                    markersize=exp_markersize, capsize=exp_capsize,
                    color=sp_colors.get(label_a, None),
                    zorder=exp_zorder, alpha=0.95,
                    label='_nolegend_'
                )
            if exp_data_b is not None and sp in exp_data_b and show_mean:
                ax.errorbar(
                    np.asarray(exp_data_b[sp]["t"]), np.asarray(exp_data_b[sp]["y"]),
                    yerr=np.asarray(exp_data_b[sp]["yerr"]),
                    fmt='o', linestyle='none',
                    markersize=exp_markersize, capsize=exp_capsize,
                    color=sp_colors.get(label_b, None),
                    zorder=exp_zorder, alpha=0.95,
                    label='_nolegend_'
                )

            ax.set_title(
                f"Time Profile of {sp} Dynamics for N = {Xa.shape[0]} Simulations",
                fontsize=20
            )
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            if si == 0:
                ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
            ax.set_xlabel('time [h]', fontsize=12)
            ax.grid(alpha=0.2, linestyle=':')

        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 6

        if legend_pos == "below":
            fig.legend(handles, labels, frameon=False, loc='lower center',
                       bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=legend_fontsize)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        elif legend_pos == "above":
            fig.legend(handles, labels, frameon=False, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=legend_fontsize)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
        elif legend_pos == 'none':
            pass
        else:
            axes[-1].legend(handles, labels, frameon=False, loc='upper right')
            fig.tight_layout()

        fig.patch.set_facecolor('white')
        fig.savefig(f'pics/AMPK_MTOR_ULK_Dynamics_{cond_label}.png', dpi=600)
        return fig

    # ---------------- LAYOUT 2: condition layout ----------------
    if separate_conditions:
        figs = []

        def _single_condition_fig(t_h, X, q10, q25, qmed, q75, q90, qmean,
                                  samp_idx, cond_label):
            fig_c, ax = plt.subplots(1, 1, figsize=figsize)
            fig_c.set_dpi(display_dpi)

            sp_colors = {}
            for si, sp in enumerate(species):
                for ridx in samp_idx:
                    ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                            label='_nolegend_')

                if show_p80:
                    ax.fill_between(t_h, q10[:, si], q90[:, si],
                                    alpha=band_alpha_p80, linewidth=0,
                                    label='_nolegend_')
                if show_iqr:
                    ax.fill_between(t_h, q25[:, si], q75[:, si],
                                    alpha=band_alpha_iqr, linewidth=0,
                                    label='_nolegend_')

                if show_mean:
                    line = ax.plot(t_h, qmean[:, si], lw=2,
                                   label=f"{sp} mean")[0]
                    sp_colors[sp] = line.get_color()

                    # --- experimental overlay (errorbars) ---
                    exp_data = exp_data_a if cond_label == label_a else exp_data_b
                    if exp_data is not None and sp in exp_data:
                        t_pts = np.asarray(exp_data[sp]["t"], dtype=float)
                        y_pts = np.asarray(exp_data[sp]["y"], dtype=float)
                        yerr  = np.asarray(exp_data[sp]["yerr"], dtype=float)
                        ax.errorbar(
                            t_pts, y_pts, yerr=yerr,
                            fmt='o', linestyle='none',
                            markersize=exp_markersize,
                            capsize=exp_capsize,
                            color=sp_colors.get(sp, None),
                            alpha=0.95,
                            zorder=exp_zorder,
                            label='_nolegend_',
                        )

                if show_median:
                    ax.plot(t_h, qmed[:, si], lw=1, linestyle='--',
                            label=f"{sp} median")

            if show_axp and set(species) == {'ATP','ADP','AMP'}:
                ax.plot(t_h, qmean.sum(axis=1), lw=2.2, linestyle=':',
                        label='AXP mean')

            title = ("Starvation" if cond_label == 'Starved' else "Basal State")
            ax.set_title(f"{user_title}", fontsize=32)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours+0.1, 6))
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_xlabel('time [h]', fontsize=30)
            ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=30)
            #ax.set_ylabel(r'concentration [mM]', fontsize=30)
            ax.grid(alpha=0.2, linestyle=':')
##########################################################################################################
            # curve labels with connectors (optional)
            if curve_labels and show_mean:
                x0, x1 = t_h[0], t_h[-1]
                if 0.0 <= curve_label_xfrac <= 1.0:
                    x_target = x0 + curve_label_xfrac * (x1 - x0)
                else:
                    x_target = curve_label_xfrac
                x_target = np.clip(x_target, x0, x1)

                y_at_x = [np.interp(x_target, t_h, qmean[:, si]) for si in range(len(species))]

                y_min, y_max = ax.get_ylim()
                y_range = max(1e-12, (y_max - y_min))
                y_spread = _spread(y_at_x, curve_label_min_gap_frac * y_range)
                y_labels = [y_s + curve_label_y_offset_frac * y_range for y_s in y_spread]

                x_label = x_target + curve_label_xpad
                if x_label > x1:
                    ax.set_xlim(x0, x_label + 0.5)

                for sp, y_lab, y_true in zip(species, y_labels, y_at_x):
                    col = sp_colors.get(sp, 'black')
                    ax.text(x_label, y_lab, sp,
                            color=col, va='center', ha='left',
                            fontsize=legend_fontsize, weight='bold')
                    if curve_label_connectors:
                        ax.plot([x_target, x_label - 0.2],
                                [y_true,  y_lab],
                                lw=2.0, alpha=0.6, color=col,
                                label='_nolegend_',
                                clip_on=False)

            handles_c, labels_c = ax.get_legend_handles_labels()
            if legend_pos == "below":
                fig_c.legend(handles_c, labels_c, frameon=False, loc='lower center',
                             bbox_to_anchor=(0.5, -0.01), ncol=max(1, len(labels_c)),
                             fontsize=legend_fontsize)
                fig_c.tight_layout(rect=[0, 0.06, 1, 1])
            elif legend_pos == "above":
                fig_c.legend(handles_c, labels_c, frameon=False, loc='upper center',
                             bbox_to_anchor=(0.5, 1.15), ncol=max(1, len(labels_c)),
                             fontsize=legend_fontsize)
                fig_c.tight_layout(rect=[0, 0, 1, 0.94])
            elif legend_pos == 'none':
                pass
            else:
                ax.legend(frameon=False, loc='upper right')
                fig_c.tight_layout()

            fig_c.patch.set_facecolor('white')
            return fig_c

        figs.append(_single_condition_fig(
            t_h_a, Xa, qa10, qa25, qamed, qa75, qa90, qamean,
            samp_a, label_a
        ))
        figs.append(_single_condition_fig(
            t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90, qbmean,
            samp_b, label_b
        ))
        for fig, cond_label in zip(figs, [label_a, label_b]):
            fig.savefig(f'pics/AMPK_MTOR_ULK_Dynamics_{cond_label}.png', dpi=600)
        return figs

    # option B: one figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.set_dpi(display_dpi)

    def _plot_condition_panel(ax, t_h, X, q10, q25, qmed, q75, q90, qmean,
                              samp_idx, cond_label, bucket, exp_data):
        first_for_stat = True
        sp_colors = {}

        for si, sp in enumerate(species):
            for ridx in samp_idx:
                ax.plot(t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                        label='_nolegend_')

            if show_p80:
                band = ax.fill_between(t_h, q10[:, si], q90[:, si],
                                       alpha=band_alpha_p80, linewidth=0)
                if first_for_stat:
                    bucket[('P80', None)] = band
            if show_iqr:
                band = ax.fill_between(t_h, q25[:, si], q75[:, si],
                                       alpha=band_alpha_iqr, linewidth=0)
                if first_for_stat:
                    bucket[('IQR', None)] = band

            if show_mean:
                line = ax.plot(t_h, qmean[:, si], lw=2)[0]
                sp_colors[sp] = line.get_color()
                bucket[('mean', sp)] = line

                # experimental overlay
                if exp_data is not None and sp in exp_data:
                    ax.errorbar(
                        np.asarray(exp_data[sp]["t"]), np.asarray(exp_data[sp]["y"]),
                        yerr=np.asarray(exp_data[sp]["yerr"]),
                        fmt='o', linestyle='none',
                        markersize=exp_markersize, capsize=exp_capsize,
                        color=sp_colors.get(sp, None),
                        zorder=exp_zorder, alpha=0.95,
                        label='_nolegend_'
                    )

            if show_median:
                line = ax.plot(t_h, qmed[:, si], lw=1, linestyle='--')[0]
                bucket[('median', sp)] = line

            first_for_stat = False

        if cond_label == 'Starved':
            ax.set_title(f"AXP Dynamics during Starvation  for N = {X.shape[0]} simulations",
                         fontsize=15)
        else:
            ax.set_title(f"AXP Dynamics in Basal State for N = {X.shape[0]} simulations",
                         fontsize=15)

        ax.set_xlim(0, total_hours)
        ax.set_xticks(np.arange(0, total_hours+0.1, 6))
        ax.set_xlabel('time [h]', fontsize=12)
        ax.grid(alpha=0.2, linestyle=':')

    _plot_condition_panel(axes[0], t_h_a, Xa, qa10, qa25, qamed, qa75, qa90,
                          qamean, samp_a, label_a, legend_data[label_a], exp_data_a)
    _plot_condition_panel(axes[1], t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90,
                          qbmean, samp_b, label_b, legend_data[label_b], exp_data_b)

    axes[0].set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
    axes[1].tick_params(labelleft=True)

    if show_axp and set(species) == {'ATP','ADP','AMP'}:
        line_a = axes[0].plot(t_h_a, qamean.sum(axis=1), lw=1.2, linestyle=':')[0]
        line_b = axes[1].plot(t_h_b, qbmean.sum(axis=1), lw=1.2, linestyle=':')[0]
        legend_data[label_a][('AXP_mean', None)] = line_a
        legend_data[label_b][('AXP_mean', None)] = line_b

    # shared legend
    if legend_pos in {"below", "above"}:
        cols = []
        if show_iqr:
            cols.append(('IQR', None))
        if show_p80:
            cols.append(('P80', None))
        for sp in species:
            if show_mean:
                cols.append(('mean', sp))
            if show_median:
                cols.append(('median', sp))
        if show_axp and set(species) == {'ATP','ADP','AMP'}:
            cols.append(('AXP_mean', None))

        def desc_str(key):
            kind, sp = key
            if kind == 'IQR':
                return 'IQR'
            if kind == 'P80':
                return '10–90%'
            if kind == 'mean' and sp is not None:
                return f'{sp} mean'
            if kind == 'median' and sp is not None:
                return f'{sp} median'
            if kind == 'AXP_mean':
                return 'AXP mean'
            return str(key)

        handles, labels = [], []
        for cond_label in (label_a, label_b):
            bucket = legend_data[cond_label]
            for key in cols:
                h = bucket.get(key)
                if h is not None:
                    handles.append(h)
                    labels.append(f"{cond_label} {desc_str(key)}")

        ncol = len(cols) if cols else 1

        if legend_pos == "below":
            fig.legend(handles, labels, frameon=False, loc='lower center',
                       bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=legend_fontsize)
            fig.tight_layout(rect=[0, 0.06, 1, 1])
        else:
            fig.legend(handles, labels, frameon=False, loc='upper center',
                       bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=legend_fontsize)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
    elif legend_pos == 'none':
        pass
    else:
        axes[-1].legend(frameon=False, loc=legend_pos)
        fig.tight_layout()

    fig.patch.set_facecolor('white')
    fig.savefig(f'pics/AMPK_MTOR_ULK_Dynamics_{cond_label}.png', dpi=600)
    return fig

def plot_compare_summary_ampk_conceptual_perturbed(
    df_dict_a, df_dict_b, label_a='cond A', label_b='cond B',
    curve_transforms=None, ylim=None, **plot_kwargs,
):
    """Plot perturbed model curves while keeping experimental whiskers unperturbed.

    ``curve_transforms`` has the form
    ``{condition_label: {species: callable(time_hours, values)}}``. The callable
    is applied separately to every simulation. ``ylim`` clips runaway curves.
    """
    import numpy as np
    import pandas as pd

    species = plot_kwargs.get('species', ('ATP', 'ADP', 'AMP'))
    species = (species,) if isinstance(species, str) else tuple(species)
    dt_minutes = plot_kwargs.get('dt_minutes', 30)
    total_hours = plot_kwargs.get('total_hours', 84)
    n_expected = int(total_hours * 60 / dt_minutes + 1)

    def _stack_runs(df_dict, species_, n_expected=None):
        species_ = (species_,) if isinstance(species_, str) else tuple(species_)
        max_idx = max(df.index.max() for df in df_dict.values())
        if n_expected is not None:
            max_idx = max(max_idx, n_expected)
        idx = pd.RangeIndex(1, max_idx + 1)

        mats = []
        for _, df in df_dict.items():
            tmp = (df.sort_index()
                     .reindex(idx)
                     .apply(pd.to_numeric, errors='coerce')
                     .ffill().bfill())
            if set(species_).issubset(tmp.columns):
                mats.append(tmp.loc[:, list(species_)].to_numpy())
        if not mats:
            raise ValueError(f"No runs contained the requested species: {species_}")
        return idx, np.stack(mats, axis=0)

    def _auto_experiment_from_mean(
        t_h: np.ndarray, qmean: np.ndarray, species_: tuple, n_points: int=8,
        time_points_h=None, noise_frac: float=0.03, err_frac: float=0.06, seed: int=0,
    ):
        rng = np.random.default_rng(seed)
        x0, x1 = float(t_h[0]), float(t_h[-1])

        if time_points_h is None:
            t_pts = np.linspace(x0 + 0.05*(x1-x0), x1 - 0.05*(x1-x0), n_points)
        else:
            t_pts = np.asarray(time_points_h, dtype=float)
            t_pts = np.clip(t_pts, x0, x1)

        exp = {}
        for si, sp in enumerate(species_):
            y_true = np.interp(t_pts, t_h, qmean[:, si])
            scale = np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            y_pts = y_true + rng.normal(0.0, noise_frac * scale)
            yerr = err_frac * np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            exp[sp] = {"t": t_pts, "y": y_pts, "yerr": yerr}
        return exp

    plot_kwargs = dict(plot_kwargs)
    if plot_kwargs.get('exp_auto', True):
        for df_dict, exp_key, seed in (
            (df_dict_a, 'exp_data_a', plot_kwargs.get('exp_seed', 7)),
            (df_dict_b, 'exp_data_b', plot_kwargs.get('exp_seed', 7) + 1),
        ):
            if plot_kwargs.get(exp_key) is None:
                idx, X = _stack_runs(df_dict, species, n_expected)
                t_h = (idx.values - 1) * dt_minutes / 60.0
                plot_kwargs[exp_key] = _auto_experiment_from_mean(
                    t_h, X.mean(axis=0), species,
                    n_points=plot_kwargs.get('exp_n_points', 9),
                    time_points_h=plot_kwargs.get('exp_time_points_h'),
                    noise_frac=plot_kwargs.get('exp_noise_frac', 0.1),
                    err_frac=plot_kwargs.get('exp_err_frac', 0.15), seed=seed,
                )
        plot_kwargs['exp_auto'] = False

    transformed = {
        label_a: {key: df.copy() for key, df in df_dict_a.items()},
        label_b: {key: df.copy() for key, df in df_dict_b.items()},
    }
    for condition, transforms in (curve_transforms or {}).items():
        if condition not in transformed:
            raise ValueError(f'Unknown condition {condition!r}; use {label_a!r} or {label_b!r}.')
        for df in transformed[condition].values():
            t_h = np.arange(len(df)) * dt_minutes / 60.0
            for sp, transform in transforms.items():
                if sp in df:
                    values = np.asarray(transform(t_h, df[sp].to_numpy(copy=True)), dtype=float)
                    if values.shape != (len(df),):
                        raise ValueError(f'Transform for {condition}/{sp} must return {len(df)} values.')
                    df[sp] = np.nan_to_num(values, posinf=1e100, neginf=-1e100)

    result = plot_compare_summary_ampk_conceptual(
        df_dict_a=transformed[label_a], df_dict_b=transformed[label_b],
        label_a=label_a, label_b=label_b, **plot_kwargs,
    )
    if ylim is not None:
        figures = result if isinstance(result, list) else [result]
        for fig in figures:
            for ax in fig.axes:
                ax.set_ylim(*ylim)
                for artist in (*ax.lines, *ax.texts):
                    artist.set_clip_on(True)
        if plot_kwargs.get('separate_conditions') and len(figures) == 2:
            for fig, condition in zip(figures, (label_a, label_b)):
                fig.savefig(f'pics/AMPK_MTOR_ULK_Dynamics_{condition}.png', dpi=600)
    return result

def add_ampk_totals(df_dict,
                    ampk_name='AMPK_t',
                    ampka_name='AMPKA_t'):
    """
    Return a new dict of DataFrames where each df has two extra columns:
    AMPK_tot  = AMPK  + AMPK_ATP  + AMPK_ADP  + AMPK_AMP
    AMPKA_tot = AMPKA + AMPKA_ATP + AMPKA_ADP + AMPKA_AMP
    """
    out = {}
    for key, df in df_dict.items():
        tmp = df.copy()

        tmp[ampk_name] = (
            tmp['AMPK'] +
            tmp['AMPK_ATP'] +
            tmp['AMPK_ADP'] +
            tmp['AMPK_AMP']
        )

        tmp[ampka_name] = (
            tmp['AMPKA'] +
            tmp['AMPKA_ATP'] +
            tmp['AMPKA_ADP'] +
            tmp['AMPKA_AMP']
        )

        out[key] = tmp
    return out

stac_ampk   = add_ampk_totals(mechs['stac'].orig_time_sim_df)
starve_ampk = add_ampk_totals(mechs['starve'].orig_time_sim_df)

figs_mau = plot_compare_summary_ampk_conceptual(
    df_dict_a=starve_ampk,
    df_dict_b=stac_ampk,
    label_a='Starved',
    label_b='Basal',
    species=('AMPKA_t', 'ULKA', 'MTORA'),
    layout_mode="condition",   # or "species", as you like
    show_iqr=False, show_p80=False,
    show_mean=True, show_median=False,
    sample_trajs=0,
    traj_alpha=0.5,
    legend_pos="none",
    separate_conditions=True, 
    curve_labels=True,
    curve_label_xfrac=40,        # 1.0 = end of time window
    curve_label_xpad=1.5,         # 1.5 h to the right of that point
    curve_label_y_offset_frac=0.08,  # 2% of y-range above the curve
    figsize=(8, 6),
    legend_fontsize=18,
    display_dpi=450,
    user_title='',#AMPK-MTOR-ULK Dynamics',
    exp_noise_frac = 0.08,       # closeness to mean (relative)
    exp_err_frac = 0.07,         # errorbar size (relative)
)


# %%
def plot_compare_summary_ampk_conceptual(
    df_dict_a, df_dict_b, label_a: str='cond A', label_b: str='cond B', species=('ATP', 'ADP', 'AMP'),
    dt_minutes=30, total_hours=84, layout_mode: str='species', show_iqr: bool=True, show_p80: bool=False,
    show_mean: bool=True, mean_lw: float=2.0, show_median: bool=False, sample_trajs: int=0, traj_alpha: float=0.1,
    band_alpha_iqr: float=0.2, band_alpha_p80: float=0.1, legend_pos: str='below', legend_inside_pos: str='top',
    legend_fontsize: float=12, legend_lw: float=3.0, show_axp: bool=True, figsize=(12, 8), display_dpi: int=150,
    curve_labels: bool=False, curve_label_min_gap_frac: float=0.02, curve_label_xpad: float=-3.0,
    curve_label_connectors: bool=True, multi_fig: bool=False, separate_conditions: bool=False, user_title='',
    curve_label_xfrac=1, curve_label_y_offset_frac=0.02, exp_data_a: dict | None=None, exp_data_b: dict | None=None,
    exp_auto: bool=True, exp_n_points: int=9, exp_time_points_h=None, exp_noise_frac: float=0.1, exp_err_frac: float=0.15,
    exp_seed: int=7, exp_markersize: float=6, exp_capsize: float=3, exp_zorder: int=6,
):
    """
    Compare conditions A vs B for the given species, with optional experimental
    datapoints (error bars) overlaid.

    Legend:
      legend_pos="inside", legend_inside_pos="top"
      legend_pos="inside", legend_inside_pos="bottom"

    When show_mean=False, one representative trajectory per species is used
    for the legend, while all other trajectories remain excluded.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---------------- helpers ----------------
    def _normalize_species(species_):
        if isinstance(species_, str):
            return (species_,)
        return tuple(species_)

    def _species_cmap(sp):
        """Use one color family per signaling species."""
        name = sp.upper()
        if 'AMPK' in name:
            return plt.get_cmap('Blues')
        if 'ULK' in name:
            return plt.get_cmap('Oranges')
        if 'MTOR' in name:
            return plt.get_cmap('Greens')
        return plt.get_cmap('Greys')

    def _species_color(sp, shade=0.85):
        """Return one shade from the species-specific color family."""
        return _species_cmap(sp)(shade)

    def _trajectory_colors(sp, n):
        """Generate different shades of the same species-specific color."""
        if n <= 0:
            return []
        if n == 1:
            return [_species_color(sp, 0.7)]

        shades = np.linspace(0.35, 0.78, n)
        shades[0] = 0.82
        return [_species_color(sp, s) for s in shades]

    def _stack_runs(df_dict, species_, n_expected=None):
        species_ = _normalize_species(species_)
        max_idx = max(df.index.max() for df in df_dict.values())

        if n_expected is not None:
            max_idx = max(max_idx, n_expected)

        idx = pd.RangeIndex(1, max_idx + 1)
        mats = []

        for _, df in df_dict.items():
            tmp = df.sort_index().reindex(idx).apply(pd.to_numeric, errors='coerce').ffill().bfill()
            if set(species_).issubset(tmp.columns):
                mats.append(tmp.loc[:, list(species_)].to_numpy())

        if not mats:
            raise ValueError(f'No runs contained the requested species: {species_}')

        return idx, np.stack(mats, axis=0)

    def _summary(X: np.ndarray):
        q10 = np.quantile(X, 0.1, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        med = np.quantile(X, 0.5, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        q90 = np.quantile(X, 0.9, axis=0)
        mean = X.mean(axis=0)
        return q10, q25, med, q75, q90, mean

    def _spread(y_values, min_gap):
        y = np.array(y_values, dtype=float)
        order = np.argsort(y)
        y_sorted = y[order]

        adj = np.empty_like(y_sorted)
        prev = -np.inf

        for i, val in enumerate(y_sorted):
            if val <= prev + min_gap:
                adj[i] = prev + min_gap
            else:
                adj[i] = val
            prev = adj[i]

        result = np.empty_like(adj)
        result[order] = adj
        return result

    def _auto_experiment_from_mean(
        t_h: np.ndarray, qmean: np.ndarray, species_: tuple, n_points: int=8,
        time_points_h=None, noise_frac: float=0.03, err_frac: float=0.06, seed: int=0,
    ):
        rng = np.random.default_rng(seed)
        x0, x1 = float(t_h[0]), float(t_h[-1])

        if time_points_h is None:
            t_pts = np.linspace(x0 + 0.05*(x1-x0), x1 - 0.05*(x1-x0), n_points)
        else:
            t_pts = np.asarray(time_points_h, dtype=float)
            t_pts = np.clip(t_pts, x0, x1)

        exp = {}

        for si, sp in enumerate(species_):
            y_true = np.interp(t_pts, t_h, qmean[:, si])
            scale = np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            y_pts = y_true + rng.normal(0.0, noise_frac * scale)
            yerr = err_frac * np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            exp[sp] = {'t': t_pts, 'y': y_pts, 'yerr': yerr}

        return exp

    def _inside_legend_kwargs():
        if legend_inside_pos == 'top':
            return {'loc': 'upper center', 'bbox_to_anchor': (0.5, 0.99)}
        elif legend_inside_pos == 'bottom':
            return {'loc': 'lower center', 'bbox_to_anchor': (0.5, 0.01)}
        else:
            raise ValueError('legend_inside_pos must be "top" or "bottom" when legend_pos="inside".')

    def _legend(owner, *args, **kwargs):
        """Create a legend and independently control the legend line width."""
        leg = owner.legend(*args, **kwargs)
        for line in leg.get_lines():
            line.set_linewidth(legend_lw)
        return leg

    # ---------------- validate / prepare ----------------
    if layout_mode not in {'species', 'condition'}:
        raise ValueError('layout_mode must be "species" or "condition".')

    species = _normalize_species(species)

    if multi_fig and len(species) > 1:
        figs = []

        for sp in species:
            figs.append(plot_compare_summary_ampk_conceptual(
                df_dict_a=df_dict_a, df_dict_b=df_dict_b, label_a=label_a, label_b=label_b, species=(sp,),
                dt_minutes=dt_minutes, total_hours=total_hours,
                layout_mode=layout_mode, show_iqr=show_iqr, show_p80=show_p80, show_mean=show_mean, mean_lw=mean_lw,
                show_median=show_median, sample_trajs=sample_trajs, traj_alpha=traj_alpha, band_alpha_iqr=band_alpha_iqr,
                band_alpha_p80=band_alpha_p80, legend_pos=legend_pos, legend_inside_pos=legend_inside_pos,
                legend_fontsize=legend_fontsize, legend_lw=legend_lw, show_axp=show_axp, figsize=figsize, display_dpi=display_dpi,
                curve_labels=curve_labels, curve_label_min_gap_frac=curve_label_min_gap_frac, curve_label_xpad=curve_label_xpad,
                curve_label_connectors=curve_label_connectors, multi_fig=False, separate_conditions=separate_conditions,
                user_title=user_title, curve_label_xfrac=curve_label_xfrac, curve_label_y_offset_frac=curve_label_y_offset_frac,
                exp_data_a=exp_data_a, exp_data_b=exp_data_b, exp_auto=exp_auto, exp_n_points=exp_n_points,
                exp_time_points_h=exp_time_points_h, exp_noise_frac=exp_noise_frac, exp_err_frac=exp_err_frac,
                exp_seed=exp_seed, exp_markersize=exp_markersize, exp_capsize=exp_capsize, exp_zorder=exp_zorder,
            ))

        return figs

    n_expected = int(total_hours * 60 / dt_minutes + 1)

    idx_a, Xa = _stack_runs(df_dict_a, species, n_expected)
    idx_b, Xb = _stack_runs(df_dict_b, species, n_expected)

    t_h_a = (idx_a.values - 1) * dt_minutes / 60.0
    t_h_b = (idx_b.values - 1) * dt_minutes / 60.0

    qa10, qa25, qamed, qa75, qa90, qamean = _summary(Xa)
    qb10, qb25, qbmed, qb75, qb90, qbmean = _summary(Xb)

    if exp_auto and exp_data_a is None:
        exp_data_a = _auto_experiment_from_mean(
            t_h=t_h_a, qmean=qamean, species_=species, n_points=exp_n_points,
            time_points_h=exp_time_points_h, noise_frac=exp_noise_frac,
            err_frac=exp_err_frac, seed=exp_seed,
        )

    if exp_auto and exp_data_b is None:
        exp_data_b = _auto_experiment_from_mean(
            t_h=t_h_b, qmean=qbmean, species_=species, n_points=exp_n_points,
            time_points_h=exp_time_points_h, noise_frac=exp_noise_frac,
            err_frac=exp_err_frac, seed=exp_seed + 1,
        )

    rng = np.random.default_rng(0)
    samp_a = rng.choice(Xa.shape[0], size=min(sample_trajs, Xa.shape[0]), replace=False) if sample_trajs else []
    samp_b = rng.choice(Xb.shape[0], size=min(sample_trajs, Xb.shape[0]), replace=False) if sample_trajs else []

    legend_data = {label_a: {}, label_b: {}}

    # ========================================================
    # LAYOUT 1: one subplot per species
    # ========================================================
    if layout_mode == 'species':
        fig, axes = plt.subplots(1, len(species), figsize=figsize, sharex=True)
        fig.set_dpi(display_dpi)

        if len(species) == 1:
            axes = [axes]

        for si, sp in enumerate(species):
            ax = axes[si]

            first_traj_a = None
            first_traj_b = None

            traj_colors_a = _trajectory_colors(sp, len(samp_a))
            traj_colors_b = _trajectory_colors(sp, len(samp_b))

            for j, ridx in enumerate(samp_a):
                line = ax.plot(
                    t_h_a, Xa[ridx, :, si], lw=0.6, alpha=traj_alpha,
                    color=traj_colors_a[j], label='_nolegend_'
                )[0]

                if j == 0:
                    first_traj_a = line

            for j, ridx in enumerate(samp_b):
                line = ax.plot(
                    t_h_b, Xb[ridx, :, si], lw=0.6, alpha=traj_alpha,
                    color=traj_colors_b[j], label='_nolegend_'
                )[0]

                if j == 0:
                    first_traj_b = line

            if show_p80:
                ax.fill_between(
                    t_h_a, qa10[:, si], qa90[:, si], alpha=band_alpha_p80,
                    linewidth=0, color=_species_color(sp, 0.55), label='_nolegend_'
                )

            if show_iqr:
                ax.fill_between(
                    t_h_a, qa25[:, si], qa75[:, si], alpha=band_alpha_iqr,
                    linewidth=0, color=_species_color(sp, 0.62), label='_nolegend_'
                )

            if show_p80:
                ax.fill_between(
                    t_h_b, qb10[:, si], qb90[:, si], alpha=band_alpha_p80,
                    linewidth=0, color=_species_color(sp, 0.55), label='_nolegend_'
                )

            if show_iqr:
                ax.fill_between(
                    t_h_b, qb25[:, si], qb75[:, si], alpha=band_alpha_iqr,
                    linewidth=0, color=_species_color(sp, 0.62), label='_nolegend_'
                )

            sp_colors = {}

            if show_mean:
                mean_color = _species_color(sp, 0.9)

                line_a = ax.plot(t_h_a, qamean[:, si], lw=mean_lw, color=mean_color, label=sp)[0]
                line_b = ax.plot(t_h_b, qbmean[:, si], lw=mean_lw, color=mean_color, label='_nolegend_')[0]

                sp_colors[label_a] = line_a.get_color()
                sp_colors[label_b] = line_b.get_color()

            elif first_traj_a is not None:
                first_traj_a.set_label(sp)
                sp_colors[label_a] = first_traj_a.get_color()

                if first_traj_b is not None:
                    sp_colors[label_b] = first_traj_b.get_color()

            if show_median:
                median_color = _species_color(sp, 0.78)
                ax.plot(t_h_a, qamed[:, si], lw=1, linestyle='--', color=median_color, label='_nolegend_')
                ax.plot(t_h_b, qbmed[:, si], lw=1, linestyle='--', color=median_color, label='_nolegend_')

            if exp_data_a is not None and sp in exp_data_a:
                ax.errorbar(
                    np.asarray(exp_data_a[sp]['t']), np.asarray(exp_data_a[sp]['y']),
                    yerr=np.asarray(exp_data_a[sp]['yerr']), fmt='o', linestyle='none',
                    markersize=exp_markersize, capsize=exp_capsize,
                    color=sp_colors.get(label_a, None), zorder=exp_zorder,
                    alpha=0.95, label='_nolegend_'
                )

            if exp_data_b is not None and sp in exp_data_b:
                ax.errorbar(
                    np.asarray(exp_data_b[sp]['t']), np.asarray(exp_data_b[sp]['y']),
                    yerr=np.asarray(exp_data_b[sp]['yerr']), fmt='o', linestyle='none',
                    markersize=exp_markersize, capsize=exp_capsize,
                    color=sp_colors.get(label_b, None), zorder=exp_zorder,
                    alpha=0.95, label='_nolegend_'
                )

            ax.set_title(f'Time Profile of {sp} Dynamics for N = {Xa.shape[0]} Simulations', fontsize=20)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours + 0.1, 6))

            if si == 0:
                ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)

            ax.set_xlabel('time [h]', fontsize=12)
            ax.grid(alpha=0.2, linestyle=':')

        handles, labels = axes[0].get_legend_handles_labels()
        ncol = max(1, len(labels))

        if legend_pos == 'below':
            _legend(
                fig, handles, labels, frameon=False, loc='lower center',
                bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=legend_fontsize
            )
            fig.tight_layout(rect=[0, 0.06, 1, 1])

        elif legend_pos == 'above':
            _legend(
                fig, handles, labels, frameon=False, loc='upper center',
                bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=legend_fontsize
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])

        elif legend_pos == 'inside':
            _legend(
                axes[-1], handles, labels, frameon=False, ncol=ncol,
                fontsize=legend_fontsize, **_inside_legend_kwargs()
            )
            fig.tight_layout()

        elif legend_pos == 'none':
            pass

        else:
            _legend(
                axes[-1], handles, labels, frameon=False, loc=legend_pos,
                ncol=ncol, fontsize=legend_fontsize
            )
            fig.tight_layout()

        fig.patch.set_facecolor('white')
        fig.savefig('pics/AMPK_MTOR_ULK_Dynamics.png', dpi=600)
        return fig

    # ========================================================
    # LAYOUT 2: condition layout, separate figures
    # ========================================================
    if separate_conditions:
        figs = []

        def _single_condition_fig(t_h, X, q10, q25, qmed, q75, q90, qmean, samp_idx, cond_label):
            fig_c, ax = plt.subplots(1, 1, figsize=figsize)
            fig_c.set_dpi(display_dpi)

            sp_colors = {}

            for si, sp in enumerate(species):
                first_traj = None
                traj_colors = _trajectory_colors(sp, len(samp_idx))

                for j, ridx in enumerate(samp_idx):
                    line = ax.plot(
                        t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                        color=traj_colors[j], label='_nolegend_'
                    )[0]

                    if j == 0:
                        first_traj = line

                if show_p80:
                    ax.fill_between(
                        t_h, q10[:, si], q90[:, si], alpha=band_alpha_p80,
                        linewidth=0, color=_species_color(sp, 0.55), label='_nolegend_'
                    )

                if show_iqr:
                    ax.fill_between(
                        t_h, q25[:, si], q75[:, si], alpha=band_alpha_iqr,
                        linewidth=0, color=_species_color(sp, 0.62), label='_nolegend_'
                    )

                if show_mean:
                    line = ax.plot(t_h, qmean[:, si], lw=mean_lw, color=_species_color(sp, 0.9), label=sp)[0]
                    sp_colors[sp] = line.get_color()

                elif first_traj is not None:
                    first_traj.set_label(sp)
                    sp_colors[sp] = first_traj.get_color()

                if show_median:
                    ax.plot(
                        t_h, qmed[:, si], lw=1, linestyle='--',
                        color=_species_color(sp, 0.78), label='_nolegend_'
                    )

                exp_data = exp_data_a if cond_label == label_a else exp_data_b

                if exp_data is not None and sp in exp_data:
                    t_pts = np.asarray(exp_data[sp]['t'], dtype=float)
                    y_pts = np.asarray(exp_data[sp]['y'], dtype=float)
                    yerr = np.asarray(exp_data[sp]['yerr'], dtype=float)

                    ax.errorbar(
                        t_pts, y_pts, yerr=yerr, fmt='o', linestyle='none',
                        markersize=exp_markersize, capsize=exp_capsize,
                        color=sp_colors.get(sp, None), alpha=0.95,
                        zorder=exp_zorder, label='_nolegend_'
                    )

            if show_axp and set(species) == {'ATP', 'ADP', 'AMP'}:
                ax.plot(t_h, qmean.sum(axis=1), lw=2.2, linestyle=':', label='_nolegend_')

            ax.set_title(f'{user_title}', fontsize=32)
            ax.set_xlim(0, total_hours)
            ax.set_xticks(np.arange(0, total_hours + 0.1, 6))
            ax.tick_params(axis='x', labelsize=26)
            ax.tick_params(axis='y', labelsize=26)
            ax.set_xlabel('time [h]', fontsize=30)
            ax.set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=35)
            ax.grid(alpha=0.2, linestyle=':')
################################################### EDIT ###################################################
            if curve_labels and show_mean:
                x0, x1 = t_h[0], t_h[-1]

                if 0.0 <= curve_label_xfrac <= 1.0:
                    x_target = x0 + curve_label_xfrac * (x1 - x0)
                else:
                    x_target = curve_label_xfrac

                x_target = np.clip(x_target, x0, x1)

                y_at_x = [
                    np.interp(x_target, t_h, qmean[:, si])
                    for si in range(len(species))
                ]

                y_min, y_max = ax.get_ylim()
                y_range = max(1e-12, y_max - y_min)

                y_spread = _spread(y_at_x, curve_label_min_gap_frac * y_range)
                y_labels = [y_s + curve_label_y_offset_frac * y_range for y_s in y_spread]

                x_label = x_target + curve_label_xpad

                if x_label > x1:
                    ax.set_xlim(x0, x_label + 0.5)

                for sp, y_lab, y_true in zip(species, y_labels, y_at_x):
                    col = sp_colors.get(sp, 'black')

                    ax.text(
                        x_label, y_lab, sp, color=col, va='center',
                        ha='left', fontsize=12, weight='bold'
                    )

                    if curve_label_connectors:
                        ax.plot(
                            [x_target, x_label - 0.2], [y_true, y_lab],
                            lw=2.0, alpha=0.6, color=col,
                            label='_nolegend_', clip_on=False
                        )

            handles_c, labels_c = ax.get_legend_handles_labels()
            ncol = max(1, len(labels_c))

            if legend_pos == 'below':
                _legend(
                    fig_c, handles_c, labels_c, frameon=False, loc='lower center',
                    bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=legend_fontsize
                )
                fig_c.tight_layout(rect=[0, 0.06, 1, 1])

            elif legend_pos == 'above':
                _legend(
                    fig_c, handles_c, labels_c, frameon=False, loc='upper center',
                    bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=legend_fontsize
                )
                fig_c.tight_layout(rect=[0, 0, 1, 0.94])

            elif legend_pos == 'inside':
                _legend(
                    ax, handles_c, labels_c, frameon=False, ncol=ncol,
                    fontsize=legend_fontsize, **_inside_legend_kwargs()
                )
                fig_c.tight_layout()

            elif legend_pos == 'none':
                pass

            else:
                _legend(
                    ax, handles_c, labels_c, frameon=False, loc=legend_pos,
                    ncol=ncol, fontsize=legend_fontsize
                )
                fig_c.tight_layout()

            fig_c.patch.set_facecolor('white')
            return fig_c

        figs.append(_single_condition_fig(
            t_h_a, Xa, qa10, qa25, qamed, qa75, qa90, qamean, samp_a, label_a
        ))
        figs.append(_single_condition_fig(
            t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90, qbmean, samp_b, label_b
        ))

        for fig, cond_label in zip(figs, [label_a, label_b]):
            fig.savefig(f'pics/AMPK_MTOR_ULK_Dynamics_{cond_label}.png', dpi=600)

        return figs

    # ========================================================
    # LAYOUT 2: condition layout, both conditions in one figure
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.set_dpi(display_dpi)

    def _plot_condition_panel(
        ax, t_h, X, q10, q25, qmed, q75, q90, qmean,
        samp_idx, cond_label, bucket, exp_data,
    ):
        first_for_stat = True

        for si, sp in enumerate(species):
            first_traj = None
            traj_colors = _trajectory_colors(sp, len(samp_idx))

            for j, ridx in enumerate(samp_idx):
                line = ax.plot(
                    t_h, X[ridx, :, si], lw=0.6, alpha=traj_alpha,
                    color=traj_colors[j], label='_nolegend_'
                )[0]

                if j == 0:
                    first_traj = line

            if show_p80:
                band = ax.fill_between(
                    t_h, q10[:, si], q90[:, si],
                    alpha=band_alpha_p80, linewidth=0,
                    color=_species_color(sp, 0.55)
                )

                if first_for_stat:
                    bucket[('P80', None)] = band

            if show_iqr:
                band = ax.fill_between(
                    t_h, q25[:, si], q75[:, si],
                    alpha=band_alpha_iqr, linewidth=0,
                    color=_species_color(sp, 0.62)
                )

                if first_for_stat:
                    bucket[('IQR', None)] = band

            if show_mean:
                line = ax.plot(t_h, qmean[:, si], lw=mean_lw, color=_species_color(sp, 0.9))[0]
                bucket[('species', sp)] = line

            elif first_traj is not None:
                bucket[('species', sp)] = first_traj

            if exp_data is not None and sp in exp_data:
                color = bucket[('species', sp)].get_color() if ('species', sp) in bucket else None

                ax.errorbar(
                    np.asarray(exp_data[sp]['t']), np.asarray(exp_data[sp]['y']),
                    yerr=np.asarray(exp_data[sp]['yerr']), fmt='o', linestyle='none',
                    markersize=exp_markersize, capsize=exp_capsize, color=color,
                    zorder=exp_zorder, alpha=0.95, label='_nolegend_'
                )

            if show_median:
                line = ax.plot(
                    t_h, qmed[:, si], lw=1, linestyle='--',
                    color=_species_color(sp, 0.78)
                )[0]
                bucket[('median', sp)] = line

            first_for_stat = False

        if cond_label == 'Starved':
            ax.set_title(f'AXP Dynamics during Starvation for N = {X.shape[0]} simulations', fontsize=15)
        else:
            ax.set_title(f'AXP Dynamics in Basal State for N = {X.shape[0]} simulations', fontsize=15)

        ax.set_xlim(0, total_hours)
        ax.set_xticks(np.arange(0, total_hours + 0.1, 6))
        ax.set_xlabel('time [h]', fontsize=12)
        ax.grid(alpha=0.2, linestyle=':')

    _plot_condition_panel(
        axes[0], t_h_a, Xa, qa10, qa25, qamed, qa75, qa90, qamean,
        samp_a, label_a, legend_data[label_a], exp_data_a
    )
    _plot_condition_panel(
        axes[1], t_h_b, Xb, qb10, qb25, qbmed, qb75, qb90, qbmean,
        samp_b, label_b, legend_data[label_b], exp_data_b
    )

    axes[0].set_ylabel(r'concentration $[\frac{mol}{cm^{3}}]$', fontsize=15)
    axes[1].tick_params(labelleft=True)

    if show_axp and set(species) == {'ATP', 'ADP', 'AMP'}:
        line_a = axes[0].plot(t_h_a, qamean.sum(axis=1), lw=1.2, linestyle=':')[0]
        line_b = axes[1].plot(t_h_b, qbmean.sum(axis=1), lw=1.2, linestyle=':')[0]

        legend_data[label_a][('AXP_mean', None)] = line_a
        legend_data[label_b][('AXP_mean', None)] = line_b

    # ---------------- shared legend ----------------
    handles, labels = [], []

    for sp in species:
        h = legend_data[label_a].get(('species', sp))

        if h is None:
            h = legend_data[label_b].get(('species', sp))

        if h is not None:
            handles.append(h)
            labels.append(sp)

    ncol = max(1, len(labels))

    if legend_pos == 'below':
        _legend(
            fig, handles, labels, frameon=False, loc='lower center',
            bbox_to_anchor=(0.5, -0.01), ncol=ncol, fontsize=legend_fontsize
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])

    elif legend_pos == 'above':
        _legend(
            fig, handles, labels, frameon=False, loc='upper center',
            bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=legend_fontsize
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

    elif legend_pos == 'inside':
        _legend(
            fig, handles, labels, frameon=False, ncol=ncol,
            fontsize=legend_fontsize, **_inside_legend_kwargs()
        )
        fig.tight_layout()

    elif legend_pos == 'none':
        pass

    else:
        _legend(
            axes[-1], handles, labels, frameon=False, loc=legend_pos,
            ncol=ncol, fontsize=legend_fontsize
        )
        fig.tight_layout()

    fig.patch.set_facecolor('white')
    fig.savefig('pics/AMPK_MTOR_ULK_Dynamics.png', dpi=600)

    return fig

figs_mau = plot_compare_summary_ampk_conceptual(
    df_dict_a=starve_ampk,
    df_dict_b=stac_ampk,
    label_a='Starved',
    label_b='Basal',
    species=('AMPKA_t', 'ULKA', 'MTORA'),
    layout_mode="condition",
    show_iqr=False,
    show_p80=False,
    show_mean=False,
    mean_lw=1.0,
    show_median=False,
    sample_trajs=100,
    traj_alpha=0.5,
    legend_pos="inside",
    legend_inside_pos="top",
    legend_fontsize=18,
    legend_lw=2.5,
    separate_conditions=True,
    curve_labels=False,
    figsize=(10, 6),
    display_dpi=450,
    user_title='',#AMPK-MTOR-ULK Dynamics',
    exp_auto=False,
    exp_noise_frac=0.08,
    exp_err_frac=0.07,
)

# %%
figs_mau[0].savefig('pics/starve_rap_combined_AMPK-mTOR-ULK_starve.png', dpi=600)
figs_mau[1].savefig('pics/starve_rap_combined_AMPK-mTOR-ULK_stac.png', dpi=600)

# %%
def plot_compare_summary_ampk_conceptual_perturbed(
    df_dict_a, df_dict_b, label_a='cond A', label_b='cond B',
    curve_transforms=None, ylim=None, **plot_kwargs,
):
    """Plot perturbed model curves while keeping experimental whiskers unperturbed.

    ``curve_transforms`` has the form
    ``{condition_label: {species: callable(time_hours, values)}}``. The callable
    is applied separately to every simulation. ``ylim`` clips runaway curves.
    """
    import numpy as np
    import pandas as pd

    species = plot_kwargs.get('species', ('ATP', 'ADP', 'AMP'))
    species = (species,) if isinstance(species, str) else tuple(species)
    dt_minutes = plot_kwargs.get('dt_minutes', 30)
    total_hours = plot_kwargs.get('total_hours', 84)
    n_expected = int(total_hours * 60 / dt_minutes + 1)

    def _stack_runs(df_dict, species_, n_expected=None):
        species_ = (species_,) if isinstance(species_, str) else tuple(species_)
        max_idx = max(df.index.max() for df in df_dict.values())
        if n_expected is not None:
            max_idx = max(max_idx, n_expected)
        idx = pd.RangeIndex(1, max_idx + 1)

        mats = []
        for _, df in df_dict.items():
            tmp = (df.sort_index()
                     .reindex(idx)
                     .apply(pd.to_numeric, errors='coerce')
                     .ffill().bfill())
            if set(species_).issubset(tmp.columns):
                mats.append(tmp.loc[:, list(species_)].to_numpy())
        if not mats:
            raise ValueError(f"No runs contained the requested species: {species_}")
        return idx, np.stack(mats, axis=0)

    def _auto_experiment_from_mean(
        t_h: np.ndarray, qmean: np.ndarray, species_: tuple, n_points: int=8,
        time_points_h=None, noise_frac: float=0.03, err_frac: float=0.06, seed: int=0,
    ):
        rng = np.random.default_rng(seed)
        x0, x1 = float(t_h[0]), float(t_h[-1])

        if time_points_h is None:
            t_pts = np.linspace(x0 + 0.05*(x1-x0), x1 - 0.05*(x1-x0), n_points)
        else:
            t_pts = np.asarray(time_points_h, dtype=float)
            t_pts = np.clip(t_pts, x0, x1)

        exp = {}
        for si, sp in enumerate(species_):
            y_true = np.interp(t_pts, t_h, qmean[:, si])
            scale = np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            y_pts = y_true + rng.normal(0.0, noise_frac * scale)
            yerr = err_frac * np.maximum(np.abs(y_true), np.max(np.abs(qmean[:, si])) * 1e-3)
            exp[sp] = {"t": t_pts, "y": y_pts, "yerr": yerr}
        return exp

    plot_kwargs = dict(plot_kwargs)
    if plot_kwargs.get('exp_auto', True):
        for df_dict, exp_key, seed in (
            (df_dict_a, 'exp_data_a', plot_kwargs.get('exp_seed', 7)),
            (df_dict_b, 'exp_data_b', plot_kwargs.get('exp_seed', 7) + 1),
        ):
            if plot_kwargs.get(exp_key) is None:
                idx, X = _stack_runs(df_dict, species, n_expected)
                t_h = (idx.values - 1) * dt_minutes / 60.0
                plot_kwargs[exp_key] = _auto_experiment_from_mean(
                    t_h, X.mean(axis=0), species,
                    n_points=plot_kwargs.get('exp_n_points', 9),
                    time_points_h=plot_kwargs.get('exp_time_points_h'),
                    noise_frac=plot_kwargs.get('exp_noise_frac', 0.1),
                    err_frac=plot_kwargs.get('exp_err_frac', 0.15), seed=seed,
                )
        plot_kwargs['exp_auto'] = False

    transformed = {
        label_a: {key: df.copy() for key, df in df_dict_a.items()},
        label_b: {key: df.copy() for key, df in df_dict_b.items()},
    }
    for condition, transforms in (curve_transforms or {}).items():
        if condition not in transformed:
            raise ValueError(f'Unknown condition {condition!r}; use {label_a!r} or {label_b!r}.')
        for df in transformed[condition].values():
            t_h = np.arange(len(df)) * dt_minutes / 60.0
            for sp, transform in transforms.items():
                if sp in df:
                    values = np.asarray(transform(t_h, df[sp].to_numpy(copy=True)), dtype=float)
                    if values.shape != (len(df),):
                        raise ValueError(f'Transform for {condition}/{sp} must return {len(df)} values.')
                    df[sp] = np.nan_to_num(values, posinf=1e100, neginf=-1e100)

    result = plot_compare_summary_ampk_conceptual(
        df_dict_a=transformed[label_a], df_dict_b=transformed[label_b],
        label_a=label_a, label_b=label_b, **plot_kwargs,
    )
    if ylim is not None:
        figures = result if isinstance(result, list) else [result]
        for fig in figures:
            for ax in fig.axes:
                ax.set_ylim(*ylim)
                for artist in (*ax.lines, *ax.texts):
                    artist.set_clip_on(True)
        if plot_kwargs.get('separate_conditions') and len(figures) == 2:
            for fig, condition in zip(figures, (label_a, label_b)):
                fig.savefig(f'pics/AMPK_MTOR_ULK_Dynamics_{condition}.png', dpi=600)
    return result