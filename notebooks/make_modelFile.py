from pathlib import Path
import ast
import re
import pandas as pd


BASE_DIR = Path("/stars/c4011027/spamms_dev")
GRID_DIR = BASE_DIR / "Outputs" / "zeta_Oph_grid"
INPUT_FILE = GRID_DIR / "input.txt"
OUTPUT_FILE = BASE_DIR / "Outputs/zeta_oph_model_index.parquet"

def strip_comment(line):
    return line.split("#", 1)[0].strip()

def parse_value(value):
    value = value.strip()
    if value == "None":
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]

def parse_input_file(input_file):
    params = {}
    with open(input_file, "r") as f:
        for raw_line in f:
            line = strip_comment(raw_line)
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = parse_value(value)
            params[key] = value
    return params


def parse_model_info(model_info_file):
    """
    Reads Model_XXXX/model_info.txt.

    Expected format:
        sigma_R:100.0
        sigma_T:100.0
        v_crit_frac:0.3
        inclination:30.0
    """
    vals = {}
    with open(model_info_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                vals[key] = float(value)
            except Exception:
                vals[key] = value
    return vals


def find_model_dirs(grid_dir):
    """
    Finds all Model_XXXX folders and sorts them by model number.
    """
    model_dirs = sorted(
        [p for p in grid_dir.glob("Model_*") if p.is_dir()],
        key=lambda p: int(re.search(r"Model_(\d+)", p.name).group(1)))
    return model_dirs


def abundance_folder_name(he, cno):
    return f"He{he}_CNO{cno}"

def build_model_index(grid_dir, input_file):
    params = parse_input_file(input_file)
    selected_lines = params.get("selected_line_list", None)
    if selected_lines is None:
        selected_lines = params.get("selected_he_line_list", None)
    if selected_lines is None:
        raise ValueError("Could not find selected_line_list in input.txt")

    selected_lines = ensure_list(selected_lines)
    times = ensure_list(params.get("times", [0.0]))
    he_abundances = ensure_list(params.get("he_abundances", [None]))
    cno_abundances = ensure_list(params.get("cno_abundances", [None]))
    model_dirs = find_model_dirs(grid_dir)

    if len(model_dirs) == 0:
        raise ValueError(f"No Model_XXXX folders found in {grid_dir}")
    rows = []

    for model_dir in model_dirs:
        model_id = model_dir.name
        model_number = int(re.search(r"Model_(\d+)", model_id).group(1))
        model_info_file = model_dir / "model_info.txt"
        if not model_info_file.exists():
            print(f"Warning: missing model_info.txt for {model_id}")
            continue
        model_info = parse_model_info(model_info_file)
        for he in he_abundances:
            for cno in cno_abundances:
                abund_dir_name = abundance_folder_name(he, cno)
                abund_dir = model_dir / abund_dir_name

                for t in times:
                    for line_name in selected_lines:
                        profile_name = f"hjd{t:.11f}_{line_name}.txt"
                        profile_path = abund_dir / profile_name
                        rows.append({
                            "model_number": model_number,
                            "model_id": model_id,
                            "model_dir": str(model_dir),
                            "model_info_path": str(model_info_file),
                            "abundance_dir": str(abund_dir),
                            "he_abundance": he,
                            "cno_abundance": cno,
                            "time": t,
                            "line_name": line_name,
                            "profile_path": str(profile_path),
                            "profile_exists": profile_path.exists(),

                            # Parameters from actual model_info.txt
                            "teff": model_info.get("teff", params.get("teff")),
                            "r_pole": model_info.get("r_pole", params.get("r_pole")),
                            "mass": model_info.get("mass", params.get("mass")),
                            "inclination": model_info.get("inclination"),
                            "v_crit_frac": model_info.get("v_crit_frac"),
                            "sigma_R": model_info.get("sigma_R"),
                            "sigma_T": model_info.get("sigma_T")})

    df = pd.DataFrame(rows)
    return df, params


def main():
    df, params = build_model_index(GRID_DIR, INPUT_FILE)
    print(df.head())
    print()
    print(f"Total rows: {len(df)}")
    print(f"Unique models: {df['model_id'].nunique()}")
    print(f"Unique lines: {df['line_name'].nunique()}")
    print(f"Missing profile files: {(~df['profile_exists']).sum()}")
    print()
    print("Parameter values in index:")
    for col in ["teff", "r_pole", "mass", "inclination", "v_crit_frac", "sigma_R", "sigma_T"]:
        if col in df.columns:
            print(f"{col}: {sorted(df[col].dropna().unique())}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print()
    print(f"Saved model index to:")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    main()
