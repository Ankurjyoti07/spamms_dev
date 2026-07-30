from pathlib import Path
import numpy as np
import pandas as pd
import json

GRID_DIR = Path("/home/c4011027/PhD_stuff/SPAMMS/Outputs/sigma_grid")
MODEL_INDEX = GRID_DIR / "model_index.parquet"
RESULT_DIR = GRID_DIR / "chisq_results" / "HEII4200_SNR300_fixed_geom"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OBS_FILES = {"HEII4200": "/home/c4011027/PhD_stuff/SPAMMS/Outputs/sigma_grid/mock_observations/mock_HEII4200_inc30_vcrit0.9_sR200_sT200_SNR300.txt"}

FIT_LINES = ["HEII4200"]
FIXED_PARAMS = {
    "teff": 40000,
    "r_pole": 6.5,
    "mass": 25,
    "inclination": 30,
    "v_crit_frac": 0.9
    }
SIGMA_MODE = "shared"   # "shared" --or-- "per_line"

OUTPUT_GLOBAL = RESULT_DIR / "chisq_results_global.csv"
OUTPUT_LINES  = RESULT_DIR / "chisq_results_lines.csv"
OUTPUT_GLOBAL = RESULT_DIR / "chisq_results_global.csv"
OUTPUT_LINES  = RESULT_DIR / "chisq_results_lines.csv"


RUN_INFO = {
    "fit_lines": FIT_LINES,
    "obs_files": OBS_FILES,
    "fixed_params": FIXED_PARAMS,
    "sigma_mode": SIGMA_MODE}

with open(RESULT_DIR / "settings.json", "w") as f:
    json.dump(RUN_INFO, f, indent=4)


############ helpers ############ 
#################################

def read_spectrum(path):
    #change this based on how observed psectra are stored
    arr = np.loadtxt(path)
    wave, flux = arr[:, 0], arr[:, 1]
    return wave, flux

def read_model_profile(path):
    arr = np.loadtxt(path)
    wave, flux = arr[:, 0], arr[:, 1]
    return wave, flux

def compute_chi2(obs_wave, obs_flux, model_wave, model_flux):
    #Computes unweighted chi-square after interpolating model onto the observed wavelength grid.
    model_interp = np.interp(obs_wave, model_wave, model_flux)
    good = np.isfinite(obs_flux) & np.isfinite(model_interp)
    chi2 = np.sum((obs_flux[good] - model_interp[good]) ** 2)
    n_pix = np.sum(good)
    return chi2, n_pix

def apply_fixed_params(df, fixed_params):
    #Apply fixed parameters:
    #    None       -> free
    #    scalar     -> fixed exactly
    #    list/tuple -> allowed values

    out = df.copy()
    for param, value in fixed_params.items():
        if value is None:
            continue
        if param not in out.columns:
            raise KeyError(f"{param} is not present in model index.")
        if isinstance(value, (list, tuple, np.ndarray)):
            out = out[out[param].isin(value)]
        else:
            out = out[np.isclose(out[param], value)]
    return out.reset_index(drop=True)

# two chisq modes: shared sigma mode
def fit_shared_sigma(df, obs_data, fit_lines):
    #shared sigma: one single value of sigma_R and sigma_T across all fitted lines.
    
    results_global = []
    results_lines = []
    model_groups = df.groupby("model_id", sort=False)
    for model_id, g_model in model_groups:
        total_chi2 = 0.0
        total_npix = 0
        valid_model = True
        first = g_model.iloc[0]
        line_records = []
        for line_name in fit_lines:
            g_line = g_model[g_model["line_name"] == line_name]
            if len(g_line) == 0:
                valid_model = False
                break
            row = g_line.iloc[0]
            if not row["profile_exists"]:
                valid_model = False
                break
            obs_wave, obs_flux = obs_data[line_name]
            model_wave, model_flux = read_model_profile(row["profile_path"])
            chi2_line, n_pix_line = compute_chi2(
                obs_wave, obs_flux,
                model_wave, model_flux)

            total_chi2 += chi2_line
            total_npix += n_pix_line
            line_records.append({
                "model_id": model_id,
                "line_name": line_name,
                "chi2_line": chi2_line,
                "n_pix_line": n_pix_line,
                "sigma_R": row["sigma_R"],
                "sigma_T": row["sigma_T"],
                "profile_path": row["profile_path"]})

        if not valid_model:
            continue
        red_chi2 = total_chi2 / max(total_npix, 1)
        results_global.append({
            "model_id": model_id,
            "chi2_total": total_chi2,
            "red_chi2": red_chi2,
            "n_pix_total": total_npix,
            "teff": first["teff"],
            "r_pole": first["r_pole"],
            "mass": first["mass"],
            "inclination": first["inclination"],
            "v_crit_frac": first["v_crit_frac"],
            "sigma_R": first["sigma_R"],
            "sigma_T": first["sigma_T"]})

        results_lines.extend(line_records)
    global_df = pd.DataFrame(results_global).sort_values("chi2_total").reset_index(drop=True)
    line_df = pd.DataFrame(results_lines)

    if len(global_df) > 0:
        rank_map = {mid: i + 1 for i, mid in enumerate(global_df["model_id"])}
        line_df["global_rank"] = line_df["model_id"].map(rank_map)
        line_df = line_df.sort_values(["global_rank", "line_name"]).reset_index(drop=True)
    return global_df, line_df


# per-line mode

def fit_per_line_sigma(df, obs_data, fit_lines):
    #global params are shared, sigma_R and sigma_T are varied per line.

    global_params = [
        "teff",
        "r_pole",
        "mass",
        "inclination",
        "v_crit_frac"]

    results_global = []
    results_lines = []
    for global_values, g_global in df.groupby(global_params, sort=False):
        global_dict = dict(zip(global_params, global_values))
        total_chi2 = 0.0
        total_npix = 0
        valid_global = True
        line_records = []
        for line_name in fit_lines:
            g_line = g_global[g_global["line_name"] == line_name]
            if len(g_line) == 0:
                valid_global = False
                break
            best = None

            for _, row in g_line.iterrows():
                if not row["profile_exists"]:
                    continue
                obs_wave, obs_flux = obs_data[line_name]
                model_wave, model_flux = read_model_profile(row["profile_path"])
                chi2_line, n_pix_line = compute_chi2(
                    obs_wave, obs_flux,
                    model_wave, model_flux)

                if best is None or chi2_line < best["chi2_line"]:
                    best = {
                        "line_name": line_name,
                        "model_id": row["model_id"],
                        "chi2_line": chi2_line,
                        "n_pix_line": n_pix_line,
                        "sigma_R": row["sigma_R"],
                        "sigma_T": row["sigma_T"],
                        "profile_path": row["profile_path"]}

            if best is None:
                valid_global = False
                break
            total_chi2 += best["chi2_line"]
            total_npix += best["n_pix_line"]
            line_records.append({**global_dict, **best})
        
        if not valid_global:
            continue
        red_chi2 = total_chi2 / max(total_npix, 1)
        results_global.append({
            **global_dict,
            "chi2_total": total_chi2,
            "red_chi2": red_chi2,
            "n_pix_total": total_npix})

        results_lines.extend(line_records)
    global_df = pd.DataFrame(results_global).sort_values("chi2_total").reset_index(drop=True)
    line_df = pd.DataFrame(results_lines)

    if len(global_df) > 0:
        global_df["global_rank"] = np.arange(1, len(global_df) + 1)
        rank_cols = global_params
        rank_lookup = global_df[rank_cols + ["global_rank"]]
        line_df = line_df.merge(rank_lookup, on=rank_cols, how="left")
        line_df = line_df.sort_values(["global_rank", "line_name"]).reset_index(drop=True)

    return global_df, line_df


# .exe
def main():

    print("Reading model index...")
    df = pd.read_parquet(MODEL_INDEX)
    print(f"Initial index rows: {len(df)}")
    print(f"Unique models: {df['model_id'].nunique()}")
    
    df = df[df["line_name"].isin(FIT_LINES)].reset_index(drop=True)
    df = apply_fixed_params(df, FIXED_PARAMS)
    print(f"Rows after filtering: {len(df)}")
    print(f"Models after filtering: {df['model_id'].nunique()}")
    print("Reading observed spectra...")
    
    obs_data = {
        line: read_spectrum(path)
        for line, path in OBS_FILES.items()
        if line in FIT_LINES}
    missing_obs = set(FIT_LINES) - set(obs_data.keys())
    if missing_obs:
        raise ValueError(f"Missing observed files for lines: {missing_obs}")
    print(f"Running chi-square search in mode: {SIGMA_MODE}")

    if SIGMA_MODE == "shared":
        global_df, line_df = fit_shared_sigma(df, obs_data, FIT_LINES)
    elif SIGMA_MODE == "per_line":
        global_df, line_df = fit_per_line_sigma(df, obs_data, FIT_LINES)
    else:
        raise ValueError("SIGMA_MODE must be either 'shared' or 'per_line'.")
    
    print()
    print("Best global fits:")
    print(global_df.head(10))
    global_df.to_csv(OUTPUT_GLOBAL, index=False)
    line_df.to_csv(OUTPUT_LINES, index=False)
    print()
    print(f"Saved global results to: {OUTPUT_GLOBAL}")
    print(f"Saved line results to:   {OUTPUT_LINES}")
    
if __name__ == "__main__":
    main()