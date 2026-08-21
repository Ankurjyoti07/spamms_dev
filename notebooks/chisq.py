from pathlib import Path
import json
import numpy as np
import pandas as pd

# Configuration
GRID_DIR = Path("/stars/c4011027/spamms_dev/Outputs/zeta_Oph_grid")
MODEL_INDEX = Path("/stars/c4011027/spamms_dev/Outputs/model_index.parquet")
OBS_FILE = Path("/stars/c4011027/spamms_dev/Outputs/spectrum.txt")

# Lines included in the joint fit
FIT_LINES = ["HEI4026", "HEII4200"]
LINE_WINDOWS = {"HEI4026": (4017.0, 4036.0),
		"HEII4200": (4192.0, 4206.0)}

# Parameters that are kept fixed during the grid search.
# Set a parameter to:
#   scalar       -> fix it to one value
#   list/tuple   -> allow only those values
#   None         -> leave the parameter free
FIXED_PARAMS = {"teff": 38000, "r_pole": 7.0, "mass": 18,
    		"inclination": None, "v_crit_frac": None}

# "shared": One common sigma_R and sigma_T combination is fitted simultaneously to all lines.
# "per_line": Each line is allowed to select its own best sigma_R and sigma_T.
SIGMA_MODE = "shared"

# Each observation gets its own output directory
RESULT_ROOT = Path( "/stars/c4011027/spamms_dev/Outputs/chisq_results")
RESULT_DIR = (RESULT_ROOT/ f"{OBS_FILE.stem}_{SIGMA_MODE}")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GLOBAL = (RESULT_DIR / "chisq_results_global.csv")
OUTPUT_LINES = (RESULT_DIR / "chisq_results_lines.csv")

RUN_INFO = {"observation_file": str(OBS_FILE), 
	    "model_index": str(MODEL_INDEX),
    	    "fit_lines": FIT_LINES, 
	    "line_windows": {line_name: list(wave_range) for line_name, wave_range in LINE_WINDOWS.items()},
            "fixed_params": FIXED_PARAMS,
	    "sigma_mode": SIGMA_MODE}

with open(RESULT_DIR / "settings.json", "w") as file:
    json.dump(RUN_INFO, file, indent=4)

# Input/output helper functions

def read_spectrum(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Observed spectrum was not found:\n{path}")
    spectrum = np.loadtxt(path)
    if spectrum.ndim != 2 or spectrum.shape[1] < 2:
        raise ValueError(
            f"The observed spectrum must contain at least "
            f"two columns:\n{path}"
        )

    wavelength = spectrum[:, 0]
    flux = spectrum[:, 1]

    return wavelength, flux


def read_model_profile(path):
    """
    Read one model line profile.
    The file must contain at least two columns: wavelength  normalized_flux
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model profile was not found:\n{path}")
    profile = np.loadtxt(path)
    if profile.ndim != 2 or profile.shape[1] < 2:
        raise ValueError(f"The model profile must contain at least two columns:\n{path}")
    wavelength = profile[:, 0]
    flux = profile[:, 1]
    return wavelength, flux


def extract_line_regions(obs_wave, obs_flux, fit_lines, line_windows):
    """
    Extract individual fitting regions from one observed spectrum.
    This works with both:
      1. the current concatenated mock spectra;
      2. future continuous observed spectra.
    returns:  obs_data as a dict {line_name: (wavelength, flux)}
    """
    obs_data = {}
    for line_name in fit_lines:
        if line_name not in line_windows:
            raise KeyError(f"No wavelength window has been defined for {line_name}.")
        wave_min, wave_max = line_windows[line_name]
        mask = (np.isfinite(obs_wave) & np.isfinite(obs_flux)
            & (obs_wave >= wave_min) & (obs_wave <= wave_max))
        if not np.any(mask):
            raise ValueError(
                f"No observed pixels were found for "
                f"{line_name} inside "
                f"{wave_min:.2f}--{wave_max:.2f} Angstrom.")
        line_wave = obs_wave[mask]
        line_flux = obs_flux[mask]
        obs_data[line_name] = (line_wave, line_flux)
    return obs_data

def compute_chi2(obs_wave, obs_flux, model_wave, model_flux, wave_min, wave_max):
    """
    Calculate the unweighted residual sum of squares between
    an observed line and a model line.
    The model is interpolated onto the observed wavelength grid.
    Only pixels inside the specified wavelength interval and inside
    the model-observation overlap are used.
    """

    obs_mask = (np.isfinite(obs_wave) & np.isfinite(obs_flux) & (obs_wave >= wave_min)
        & (obs_wave <= wave_max))
    model_mask = (np.isfinite(model_wave) & np.isfinite(model_flux)
        & (model_wave >= wave_min) & (model_wave <= wave_max))
    obs_wave_cut = obs_wave[obs_mask]
    obs_flux_cut = obs_flux[obs_mask]
    model_wave_cut = model_wave[model_mask]
    model_flux_cut = model_flux[model_mask]

    if len(obs_wave_cut) == 0:
        raise ValueError(f"No observed pixels exist inside "
            f"{wave_min:.2f}--{wave_max:.2f} Angstrom.")
    if len(model_wave_cut) < 2:
        raise ValueError(f"Fewer than two model pixels exist inside "
            f"{wave_min:.2f}--{wave_max:.2f} Angstrom.")
    # Only retain observed pixels covered by the model.
    # This prevents np.interp from using constant edge values
    # outside the model wavelength range.
    overlap_mask = ( (obs_wave_cut >= model_wave_cut.min())
        & (obs_wave_cut <= model_wave_cut.max()))
    obs_wave_cut = obs_wave_cut[overlap_mask]
    obs_flux_cut = obs_flux_cut[overlap_mask]

    if len(obs_wave_cut) == 0:
        raise ValueError(
            "The observed and model wavelength grids do not overlap.")
    model_interp = np.interp(obs_wave_cut, model_wave_cut, model_flux_cut)
    residual = obs_flux_cut - model_interp
    chi2 = np.sum(residual**2)
    n_pix = len(residual)
    return chi2, n_pix


def apply_fixed_params(dataframe, fixed_params):
    """
    Apply fixed or restricted model parameters.
    Parameter choices
    None: The parameter is left free.
    Scalar: The parameter is fixed to that value.
    list, tuple or array: Only the specified values are retained.
    """
    output = dataframe.copy()
    for parameter, value in fixed_params.items():
        if value is None:
            continue

        if parameter not in output.columns:
            raise KeyError(f"'{parameter}' is not present in the model index.")
        if isinstance(value, (list, tuple, np.ndarray)):
            allowed_values = np.asarray(value, dtype=float)
            parameter_values = output[parameter].to_numpy(dtype=float)
            keep = np.any(np.isclose(parameter_values[:, None], allowed_values[None, :]), axis=1)
            output = output[keep]

        else:
            output = output[np.isclose(output[parameter], value)]
    return output.reset_index(drop=True)


def fit_shared_sigma(
    dataframe,
    obs_data,
    fit_lines,
    line_windows,
):
    """
    Fit one shared sigma_R and sigma_T combination to all lines.

    A grid model is accepted only when every requested line exists.
    The global statistic is the sum of the statistics from all lines.
    """

    results_global = []
    results_lines = []

    model_groups = dataframe.groupby(
        "model_id",
        sort=False,
    )

    for model_id, model_group in model_groups:

        total_chi2 = 0.0
        total_npix = 0

        valid_model = True
        line_records = []

        first_row = model_group.iloc[0]

        for line_name in fit_lines:

            line_group = model_group[
                model_group["line_name"] == line_name
            ]

            if len(line_group) == 0:
                valid_model = False
                break

            row = line_group.iloc[0]

            if not bool(row["profile_exists"]):
                valid_model = False
                break

            obs_wave, obs_flux = obs_data[line_name]

            model_wave, model_flux = (
                read_model_profile(
                    row["profile_path"]
                )
            )

            wave_min, wave_max = (
                line_windows[line_name]
            )

            chi2_line, n_pix_line = compute_chi2(
                obs_wave=obs_wave,
                obs_flux=obs_flux,
                model_wave=model_wave,
                model_flux=model_flux,
                wave_min=wave_min,
                wave_max=wave_max,
            )

            total_chi2 += chi2_line
            total_npix += n_pix_line

            line_records.append(
                {
                    "model_id": model_id,
                    "line_name": line_name,
                    "chi2_line": chi2_line,
                    "mean_squared_residual_line": (
                        chi2_line
                        / max(n_pix_line, 1)
                    ),
                    "n_pix_line": n_pix_line,
                    "wave_min": wave_min,
                    "wave_max": wave_max,
                    "sigma_R": row["sigma_R"],
                    "sigma_T": row["sigma_T"],
                    "profile_path": row[
                        "profile_path"
                    ],
                }
            )

        if not valid_model:
            continue

        mean_squared_residual = (
            total_chi2 / max(total_npix, 1)
        )

        results_global.append(
            {
                "model_id": model_id,
                "chi2_total": total_chi2,
                "red_chi2": mean_squared_residual,
                "n_pix_total": total_npix,
                "teff": first_row["teff"],
                "r_pole": first_row["r_pole"],
                "mass": first_row["mass"],
                "inclination": first_row[
                    "inclination"
                ],
                "v_crit_frac": first_row[
                    "v_crit_frac"
                ],
                "sigma_R": first_row["sigma_R"],
                "sigma_T": first_row["sigma_T"],
            }
        )

        results_lines.extend(line_records)

    if len(results_global) == 0:
        return pd.DataFrame(), pd.DataFrame()

    global_df = pd.DataFrame(results_global)

    global_df = global_df.sort_values(
        "chi2_total"
    ).reset_index(drop=True)

    global_df["global_rank"] = (
        np.arange(len(global_df)) + 1
    )

    line_df = pd.DataFrame(results_lines)

    rank_map = dict(
        zip(
            global_df["model_id"],
            global_df["global_rank"],
        )
    )

    line_df["global_rank"] = (
        line_df["model_id"].map(rank_map)
    )

    line_df = line_df.sort_values(
        ["global_rank", "line_name"]
    ).reset_index(drop=True)

    return global_df, line_df


# =========================================================
# Per-line sigma fitting
# =========================================================

def fit_per_line_sigma(
    dataframe,
    obs_data,
    fit_lines,
    line_windows,
):
    """
    Fit sigma_R and sigma_T independently for each line.

    The global stellar and geometrical parameters remain shared,
    but each spectral line selects its own best sigma_R and sigma_T.
    """

    global_params = [
        "teff",
        "r_pole",
        "mass",
        "inclination",
        "v_crit_frac",
    ]

    results_global = []
    results_lines = []

    global_groups = dataframe.groupby(
        global_params,
        sort=False,
    )

    for global_values, global_group in global_groups:

        global_dictionary = dict(
            zip(
                global_params,
                global_values,
            )
        )

        total_chi2 = 0.0
        total_npix = 0

        valid_global_model = True
        line_records = []

        for line_name in fit_lines:

            line_group = global_group[
                global_group["line_name"]
                == line_name
            ]

            if len(line_group) == 0:
                valid_global_model = False
                break

            obs_wave, obs_flux = obs_data[line_name]

            wave_min, wave_max = (
                line_windows[line_name]
            )

            best_line_fit = None

            for _, row in line_group.iterrows():

                if not bool(row["profile_exists"]):
                    continue

                model_wave, model_flux = (
                    read_model_profile(
                        row["profile_path"]
                    )
                )

                chi2_line, n_pix_line = (
                    compute_chi2(
                        obs_wave=obs_wave,
                        obs_flux=obs_flux,
                        model_wave=model_wave,
                        model_flux=model_flux,
                        wave_min=wave_min,
                        wave_max=wave_max,
                    )
                )

                if (
                    best_line_fit is None
                    or chi2_line
                    < best_line_fit["chi2_line"]
                ):
                    best_line_fit = {
                        "line_name": line_name,
                        "model_id": row["model_id"],
                        "chi2_line": chi2_line,
                        "mean_squared_residual_line": (
                            chi2_line
                            / max(n_pix_line, 1)
                        ),
                        "n_pix_line": n_pix_line,
                        "wave_min": wave_min,
                        "wave_max": wave_max,
                        "sigma_R": row["sigma_R"],
                        "sigma_T": row["sigma_T"],
                        "profile_path": row[
                            "profile_path"
                        ],
                    }

            if best_line_fit is None:
                valid_global_model = False
                break

            total_chi2 += best_line_fit[
                "chi2_line"
            ]

            total_npix += best_line_fit[
                "n_pix_line"
            ]

            line_records.append(
                {
                    **global_dictionary,
                    **best_line_fit,
                }
            )

        if not valid_global_model:
            continue

        mean_squared_residual = (
            total_chi2 / max(total_npix, 1)
        )

        results_global.append(
            {
                **global_dictionary,
                "chi2_total": total_chi2,
                "red_chi2": mean_squared_residual,
                "n_pix_total": total_npix,
            }
        )

        results_lines.extend(line_records)

    if len(results_global) == 0:
        return pd.DataFrame(), pd.DataFrame()

    global_df = pd.DataFrame(results_global)

    global_df = global_df.sort_values(
        "chi2_total"
    ).reset_index(drop=True)

    global_df["global_rank"] = (
        np.arange(len(global_df)) + 1
    )

    line_df = pd.DataFrame(results_lines)

    rank_lookup = global_df[
        global_params + ["global_rank"]
    ]

    line_df = line_df.merge(
        rank_lookup,
        on=global_params,
        how="left",
    )

    line_df = line_df.sort_values(
        ["global_rank", "line_name"]
    ).reset_index(drop=True)

    return global_df, line_df


# =========================================================
# Main execution
# =========================================================

def main():

    print("Reading model index...")

    if not MODEL_INDEX.exists():
        raise FileNotFoundError(
            f"Model index was not found:\n"
            f"{MODEL_INDEX}"
        )

    dataframe = pd.read_parquet(
        MODEL_INDEX
    )

    print(
        f"Initial index rows: "
        f"{len(dataframe)}"
    )

    print(
        f"Unique models: "
        f"{dataframe['model_id'].nunique()}"
    )

    # Keep only the requested spectral lines
    dataframe = dataframe[
        dataframe["line_name"].isin(FIT_LINES)
    ].reset_index(drop=True)

    # Apply fixed stellar and geometrical parameters
    dataframe = apply_fixed_params(
        dataframe,
        FIXED_PARAMS,
    )

    print(
        f"Rows after filtering: "
        f"{len(dataframe)}"
    )

    print(
        f"Models after filtering: "
        f"{dataframe['model_id'].nunique()}"
    )

    if len(dataframe) == 0:
        raise RuntimeError(
            "No grid models remain after filtering. "
            "Check FIXED_PARAMS and FIT_LINES."
        )

    available_lines = set(
        dataframe["line_name"].unique()
    )

    missing_grid_lines = (
        set(FIT_LINES) - available_lines
    )

    if missing_grid_lines:
        raise ValueError(
            "The following requested lines are absent "
            f"from the filtered grid: "
            f"{sorted(missing_grid_lines)}"
        )

    # -----------------------------------------------------
    # Read the complete observed spectrum once
    # -----------------------------------------------------

    print()
    print("Reading observed spectrum...")

    obs_wave, obs_flux = read_spectrum(
        OBS_FILE
    )

    print(
        f"Total observed pixels: "
        f"{len(obs_wave)}"
    )

    # -----------------------------------------------------
    # Extract the requested line regions
    # -----------------------------------------------------

    obs_data = extract_line_regions(
        obs_wave=obs_wave,
        obs_flux=obs_flux,
        fit_lines=FIT_LINES,
        line_windows=LINE_WINDOWS,
    )

    print()
    print("Observed fitting regions:")

    for line_name in FIT_LINES:

        line_wave, line_flux = (
            obs_data[line_name]
        )

        wave_min, wave_max = (
            LINE_WINDOWS[line_name]
        )

        print(
            f"  {line_name}: "
            f"{len(line_wave)} pixels, "
            f"{wave_min:.2f}--"
            f"{wave_max:.2f} Angstrom"
        )

    # -----------------------------------------------------
    # Run the grid search
    # -----------------------------------------------------

    print()
    print(
        f"Running chi-square search "
        f"in mode: {SIGMA_MODE}"
    )

    if SIGMA_MODE == "shared":

        global_df, line_df = (
            fit_shared_sigma(
                dataframe=dataframe,
                obs_data=obs_data,
                fit_lines=FIT_LINES,
                line_windows=LINE_WINDOWS,
            )
        )

    elif SIGMA_MODE == "per_line":

        global_df, line_df = (
            fit_per_line_sigma(
                dataframe=dataframe,
                obs_data=obs_data,
                fit_lines=FIT_LINES,
                line_windows=LINE_WINDOWS,
            )
        )

    else:
        raise ValueError(
            "SIGMA_MODE must be either "
            "'shared' or 'per_line'."
        )

    if global_df.empty:
        raise RuntimeError(
            "The fit produced no valid results. "
            "Check the model paths, profile_exists "
            "values and requested lines."
        )

    # -----------------------------------------------------
    # Display and save results
    # -----------------------------------------------------

    print()
    print("Best global fits:")
    print(
        global_df.head(10).to_string(
            index=False
        )
    )

    global_df.to_csv(
        OUTPUT_GLOBAL,
        index=False,
    )

    line_df.to_csv(
        OUTPUT_LINES,
        index=False,
    )

    print()
    print(
        f"Saved global results to:\n"
        f"{OUTPUT_GLOBAL}"
    )

    print()
    print(
        f"Saved line results to:\n"
        f"{OUTPUT_LINES}"
    )


if __name__ == "__main__":
    main()
