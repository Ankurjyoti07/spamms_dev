import sys
from pathlib import Path
sys.path.append('/home/c4011027/PhD_stuff/SPAMMS')
import spamms as sp
from tempfile import TemporaryDirectory
from functools import partial
from multiprocessing import Pool
import subprocess
import sys
import re
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

# Configuration
SPAMMS_DIR = Path("/home/c4011027/PhD_stuff/SPAMMS")
SPAMMS_SCRIPT = SPAMMS_DIR / "spamms.py"
INPUT_TEMPLATE = SPAMMS_DIR / "input.txt"
RESULTS_DIR = SPAMMS_DIR / "Outputs/bayesian_test"
TEMP_ROOT = SPAMMS_DIR / "Outputs/bayesian_temp"

LINE_NAME = "HEII4200"
SNR = 300.0
FIXED_INCLINATION = 87
FIXED_VCRIT_FRAC = 0.83
SIGMA_R_BOUNDS = (0.0, 300.0)
SIGMA_T_BOUNDS = (0.0, 300.0)
TRUE_SIGMA_R = 149.0
TRUE_SIGMA_T = 197.0

NDIM = 2
NWALKERS = 4
NSTEPS = 50
NCORES = 4

INITIAL_GUESS = np.array([200.0, 250.0])
INITIAL_SPREAD = np.array([75.0, 75.0])


# Observation
def read_data(obsfile):
    data = np.loadtxt(obsfile)
    wave, flux = data[:, 0], data[:, 1]
    return wave, flux

# SPAMMS input
def replace_parameter(text, name, value):
    pattern = rf"^(\s*{re.escape(name)}\s*=\s*).*$"
    new_text, count = re.subn(pattern, lambda m: m.group(1) + str(value), text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Could not uniquely replace '{name}' in the input file.")
    return new_text


def build_input(template, sigma_R, sigma_T, output_dir):
    replacements = {
        "output_directory": f"{Path(output_dir).resolve()}/",
        "inclination": f"[{FIXED_INCLINATION}]",
        "v_crit_frac": f"[{FIXED_VCRIT_FRAC}]",
        "sigma_R": f"[{sigma_R}]",
        "sigma_T": f"[{sigma_T}]",
        "selected_line_list": f"['{LINE_NAME}']"}
    for name, value in replacements.items():
        template = replace_parameter(template, name, value)
    return template

# Run SPAMMS model
def run_spamms(sigma_R, sigma_T, base_template):
    with TemporaryDirectory(dir=TEMP_ROOT, prefix="spamms_") as temp_dir:
        temp_dir = Path(temp_dir).resolve()
        input_file = temp_dir / "input.txt"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        input_text = build_input(base_template, sigma_R, sigma_T, output_dir)
        input_file.write_text(input_text, encoding="utf-8")
        result = subprocess.run([sys.executable, str(SPAMMS_SCRIPT), "-i", str(input_file)], cwd=SPAMMS_DIR, stdout=subprocess.DEVNULL,stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        model_files = list(output_dir.rglob(f"hjd*_{LINE_NAME}.txt"))
        if len(model_files) != 1:
            raise FileNotFoundError(f"Expected one model, found {len(model_files)} in {output_dir}")
        model = np.loadtxt(model_files[0])
        model_wave, model_flux = model[:, 0], model[:, 1]
    return model_wave, model_flux


# Bayesian model

def log_prior(theta):
    sigma_R, sigma_T = theta
    if (SIGMA_R_BOUNDS[0] < sigma_R < SIGMA_R_BOUNDS[1]
        and SIGMA_T_BOUNDS[0] < sigma_T < SIGMA_T_BOUNDS[1]):
        return 0.0
    return -np.inf


def log_likelihood(theta, data_wave, data_flux, flux_err, base_template):
    sigma_R, sigma_T = theta
    model_wave, model_flux = run_spamms(sigma_R, sigma_T, base_template)
    f = interp1d(model_wave, model_flux, bounds_error=False, fill_value="extrapolate")
    model_interp_flux = f(data_wave)

    return -0.5 * np.sum(((data_flux - model_interp_flux) / flux_err) ** 2+ np.log(2.0 * np.pi * flux_err**2))

def log_probability(theta, data_wave, data_flux, flux_err, base_template):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data_wave, data_flux, flux_err, base_template)

# Results
def save_results(sampler):
    chain = sampler.get_chain()
    samples = sampler.get_chain(discard=NSTEPS // 2,flat=True)
    np.save(RESULTS_DIR / "posterior_samples.npy",samples,)
    np.savetxt(RESULTS_DIR / "posterior_samples.txt",samples, header="sigma_R sigma_T")
    sigma_R = np.percentile(samples[:, 0],[16, 50, 84])
    sigma_T = np.percentile(samples[:, 1],[16, 50, 84])
    acceptance = np.mean(sampler.acceptance_fraction)

    summary = (
        "Recovered parameters\n"
        "====================\n\n"
        f"sigma_R = {sigma_R[1]:.3f} "
        f"-{sigma_R[1] - sigma_R[0]:.3f} "
        f"+{sigma_R[2] - sigma_R[1]:.3f} km/s\n"
        f"sigma_T = {sigma_T[1]:.3f} "
        f"-{sigma_T[1] - sigma_T[0]:.3f} "
        f"+{sigma_T[2] - sigma_T[1]:.3f} km/s\n\n"
        f"Mean acceptance fraction = {acceptance:.3f}\n")

    (RESULTS_DIR / "summary.txt").write_text(summary,encoding="utf-8")
    print("\n" + summary)
    return chain, samples


def make_plots(chain, samples):
    labels = [r"$\sigma_R$ (km s$^{-1}$)",r"$\sigma_T$ (km s$^{-1}$)"]
    fig, axes = plt.subplots(2,1,figsize=(9, 6),sharex=True)

    for i, ax in enumerate(axes):

        ax.plot(chain[:, :, i],color="black",alpha=0.25)
        ax.axhline([TRUE_SIGMA_R, TRUE_SIGMA_T][i],color="red",linestyle="--")
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step")
    
    fig.savefig(RESULTS_DIR / "trace_plot.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    fig = corner.corner(samples,labels=labels,truths=[TRUE_SIGMA_R, TRUE_SIGMA_T],quantiles=[0.16, 0.50, 0.84],show_titles=True)
    fig.savefig(RESULTS_DIR / "corner_plot.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


# Main
def main():
    RESULTS_DIR.mkdir( parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    data_wave, data_flux = read_data('/home/c4011027/PhD_stuff/SPAMMS/Outputs/obs_profiles/Model_0003/He0.1_CNO7.5/hjd0.00000000000_HEII4200.txt') # change this
    data_wave = np.asarray(data_wave)
    data_flux = np.asarray(data_flux)
    flux_err = np.full_like( data_flux, fill_value=1.0 / SNR, dtype=float)
    base_template = INPUT_TEMPLATE.read_text(encoding="utf-8")
    rng = np.random.default_rng(1234)

    initial_positions = (INITIAL_GUESS + INITIAL_SPREAD* rng.normal(size=(NWALKERS, NDIM)))
    log_prob = partial( log_probability, data_wave=data_wave, data_flux=data_flux, flux_err=flux_err, base_template=base_template)
    backend = emcee.backends.HDFBackend( RESULTS_DIR / "chain.h5")
    backend.reset( NWALKERS, NDIM)

    with Pool(NCORES) as pool:
        sampler = emcee.EnsembleSampler( NWALKERS, NDIM, log_prob, pool=pool, backend=backend)

        for _ in tqdm( sampler.sample( initial_positions, iterations=NSTEPS), total=NSTEPS, desc="Bayesian SPAMMS fit"):
            pass
    chain, samples = save_results(sampler)
    make_plots(chain, samples)
if __name__ == "__main__":
    main()