import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _station_name(path):
    name = os.path.basename(path)
    return name.replace("PRSA_Data_", "").rsplit("_", 1)[0]


def _load_station_csv(path, covariates):
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day, hour=df.hour)
    )

    df = df[df["PM2.5"].notna()].copy()
    df[covariates] = df[covariates].ffill().bfill()
    df = df[df[covariates].notna().all(axis=1)].copy()

    df["Yresp"] = np.log(df["PM2.5"].values + 1.0)
    df["sinh"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cosh"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df


def load_beijing_blocks(data_dir, covariates, split_date, central_station, winsor_limit=5.0):
    """Prepare Beijing station-level train/test blocks."""
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "PRSA_Data_*.csv")))

    if not files:
        raise FileNotFoundError(
            f"No Beijing station CSV files found in {data_dir}."
        )

    files = sorted(
        files,
        key=lambda f: (_station_name(f) != central_station, _station_name(f)),
    )

    stations = [(_station_name(f), _load_station_csv(f, covariates)) for f in files]
    split_date = pd.Timestamp(split_date)

    train_blocks = []
    test_blocks = []

    for name, df in stations:
        tr = df[df["dt"] < split_date]
        te = df[df["dt"] >= split_date]

        Xtr_raw = np.column_stack(
            [tr[c].values for c in covariates] + [tr["sinh"].values, tr["cosh"].values]
        )
        Xte_raw = np.column_stack(
            [te[c].values for c in covariates] + [te["sinh"].values, te["cosh"].values]
        )

        mu = Xtr_raw[:, :len(covariates)].mean(axis=0)
        sd = Xtr_raw[:, :len(covariates)].std(axis=0)
        sd[sd == 0] = 1.0

        Xtr = Xtr_raw.copy()
        Xte = Xte_raw.copy()

        Xtr[:, :len(covariates)] = (Xtr_raw[:, :len(covariates)] - mu) / sd
        Xte[:, :len(covariates)] = (Xte_raw[:, :len(covariates)] - mu) / sd

        Xtr[:, :len(covariates)] = np.clip(
            Xtr[:, :len(covariates)], -winsor_limit, winsor_limit
        )
        Xte[:, :len(covariates)] = np.clip(
            Xte[:, :len(covariates)], -winsor_limit, winsor_limit
        )

        train_blocks.append((name, Xtr, tr["Yresp"].values))
        test_blocks.append((name, Xte, te["Yresp"].values))

    covariate_names = [
        "Temperature",
        "Dew point",
        "Pressure",
        "Wind speed",
        "Precipitation",
        "sin(2pi h/24)",
        "cos(2pi h/24)",
    ]

    return train_blocks, test_blocks, covariate_names
