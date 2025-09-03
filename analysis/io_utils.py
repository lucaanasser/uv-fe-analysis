import pandas as pd
import numpy as np

REQUIRED_FE_COLUMNS = {"experiment", "organism", "replicate", "treatment", "colonies", "dilution_log", "plated_uL"}
REQUIRED_UVC_COLUMNS = {"experiment", "organism", "replicate", "dose_J_m2", "colonies", "dilution_log", "plated_uL"}


def compute_cfu(colonies, dilution_log, plated_uL):
    """Compute CFU per mL.

    Args:
        colonies: Number of counted colonies (array-like or scalar).
        dilution_log: Log10 dilution factor (positive if serial dilutions were performed).
        plated_uL: Plated volume in microlitres.

    Returns:
        Corresponding CFU/mL (array-like or scalar).
    """
    dilution_factor = 10 ** np.abs(dilution_log)
    plated_mL = plated_uL / 1000.0
    return colonies / plated_mL * dilution_factor


def load_fe_data(csv_path: str) -> pd.DataFrame:
    """Load Fe experiment CSV, validate required columns and compute CFU/mL.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with additional column CFU_per_mL.
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_FE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Fe CSV: {missing}")
    df["CFU_per_mL"] = compute_cfu(df["colonies"], df["dilution_log"], df["plated_uL"])
    return df


def load_uvc_data(csv_path: str) -> pd.DataFrame:
    """Load UVC experiment CSV, validate required columns and compute CFU/mL.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with additional column CFU_per_mL.
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_UVC_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in UVC CSV: {missing}")
    df["CFU_per_mL"] = compute_cfu(df["colonies"], df["dilution_log"], df["plated_uL"])
    return df
