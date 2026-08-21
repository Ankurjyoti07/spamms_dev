from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("/stars/c4011027/spamms_dev")
GRID_DIR = BASE_DIR / "Outputs" / "spec_grid"
MODEL_INDEX = GRID_DIR / "model_index.parquet"
MOCK_DIR = GRID_DIR / "mock_observations"
MOCK_DIR.mkdir(exist_ok=True)

LINE_NAME = "HEII4200"

# choose mock input params
TRUE_PARAMS = {
    "inclination": 90,
    "v_crit_frac": 0.9,
    "sigma_R": 200,
    "sigma_T": 250}
SNR_VALUES = [100, 200, 300, 400]
RANDOM_SEED = 42

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_parquet(MODEL_INDEX)
    d = df[df["line_name"] == LINE_NAME].copy()
    for key, value in TRUE_PARAMS.items():
        d = d[np.isclose(d[key], value)]
    if len(d) == 0:
        raise ValueError("No matching model found for TRUE_PARAMS.")
    row = d.iloc[0]
    print("Using mock truth model:")
    print(row[[
        "model_id", "line_name", "inclination", "v_crit_frac",
        "sigma_R", "sigma_T", "profile_path"]])

    arr = np.loadtxt(row["profile_path"])
    wave = arr[:, 0]
    flux = arr[:, 1]
    for snr in SNR_VALUES:
        noise_sigma = 1.0 / snr
        noisy_flux = flux + rng.normal(0.0, noise_sigma, size=flux.size)
        out = np.column_stack([wave, noisy_flux])
        out_file = MOCK_DIR / (
            f"mock_{LINE_NAME}_"
            f"inc{TRUE_PARAMS['inclination']}_"
            f"vcrit{TRUE_PARAMS['v_crit_frac']}_"
            f"sR{TRUE_PARAMS['sigma_R']}_"
            f"sT{TRUE_PARAMS['sigma_T']}_"
            f"SNR{snr}.txt")
        np.savetxt(out_file, out, fmt="%.12e")
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
