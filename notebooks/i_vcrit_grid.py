import glob, sys, copy
sys.path.append('/home/c4011027/PhD_stuff/SPAMMS')
import numpy as np
import scipy.optimize as so
import astropy.constants as c
import spamms as sp
import matplotlib.pyplot as plt
import phoebe
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib import cm, colors
from pprint import pprint 
from scipy.special import lpmv
from scipy.special import sph_harm
from astropy.constants import R_sun, M_sun, G
from phoebe import u
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.special import lpmv
plt.rcParams['figure.dpi'] = 100
import pandas as pd

def inclination_rotation_sigma_grid_study(input_file, output_csv, inclinations=np.arange(10, 91, 5),
    vcrit_list=np.arange(0.0, 1.01, 0.05), sigma_R=200.0, sigma_T=200.0, n_realizations=100, seed=None):

    rng = np.random.default_rng(seed)

    fit_param_values, abund_param_values, line_list, io_dict = sp.read_input_file(input_file)
    times, obs_specs = sp.get_obs_spec_and_times(io_dict)

    rows = []

    for inc in inclinations:
        for vcrit in vcrit_list:
            print(f"Running inclination = {inc:.1f}, v_crit_frac = {vcrit:.2f}")

            fit_params_this = copy.deepcopy(fit_param_values)
            fit_params_this["inclination"] = [float(inc)]
            fit_params_this["v_crit_frac"] = [float(vcrit)]
            fit_params_this["sigma_R"] = [float(sigma_R)]
            fit_params_this["sigma_T"] = [float(sigma_T)]

            run_dictionary = sp.create_runs_and_ids(fit_params_this)[0]
            s = sp.run_sb_phoebe_model(times, abund_param_values, io_dict, run_dictionary)
            phcb = s['%09.6f' % s['times@dataset@lc'].value]

            mus = phcb['mesh@primary@mesh01@mus'].get_value()
            viss = phcb['visibilities@primary'].get_value()
            inds = (viss > 0)

            sigma_vR_vals, sigma_vT_vals = [], []

            for _ in range(n_realizations):
                sigma_R_draw = rng.normal(0.0, run_dictionary["sigma_R"], size=mus.shape[0])
                v_R = mus * sigma_R_draw

                sigma_T_draw = rng.normal(0.0, run_dictionary["sigma_T"], size=mus.shape[0])
                theta_T = rng.uniform(0.0, 2.0*np.pi, size=mus.shape[0])
                theta_mu = np.arccos(np.clip(mus, -1.0, 1.0))
                v_T = sigma_T_draw * np.sin(theta_mu) * np.cos(theta_T)

                sigma_vR_vals.append(np.std(v_R[inds]))
                sigma_vT_vals.append(np.std(v_T[inds]))

            rows.append({
                "inclination": inc,
                "v_crit_frac": vcrit,
                "sigma_R_input": run_dictionary["sigma_R"],
                "sigma_T_input": run_dictionary["sigma_T"],
                "n_realizations": n_realizations,
                "mean_sigma_vR_vis": np.mean(sigma_vR_vals),
                "std_sigma_vR_vis": np.std(sigma_vR_vals),
                "mean_sigma_vT_vis": np.mean(sigma_vT_vals),
                "std_sigma_vT_vis": np.std(sigma_vT_vals)
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"\nSaved averaged grid results to:\n{output_csv}")


if __name__ == "__main__":

    inclination_rotation_sigma_grid_study(
        input_file="/home/c4011027/PhD_stuff/SPAMMS/notebooks/input_macro_test.txt",
        output_csv="/home/c4011027/PhD_stuff/SPAMMS/Outputs/inclination_vcrit_sigma_grid.csv",
        inclinations=np.arange(5, 91, 5),
        vcrit_list=np.arange(0.1, 0.99, 0.1),
        sigma_R=200.0,
        sigma_T=200.0,
        n_realizations=200,
        seed=42)
