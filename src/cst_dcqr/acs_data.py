import numpy as np
import pandas as pd


FIPS = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "DC", 12: "Florida",
    13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana",
    19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine",
    24: "Maryland", 25: "Massachusetts", 26: "Michigan", 27: "Minnesota",
    28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska",
    32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico",
    36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio",
    40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island",
    45: "South Carolina", 46: "South Dakota", 47: "Tennessee", 48: "Texas",
    49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
}


def load_acs_blocks(data_path, n_per_state, central_state, seed, winsor_limit=5.0):
    """Prepare ACS state-level train/test blocks."""
    rng = np.random.default_rng(seed)

    df = pd.read_pickle(data_path)

    df = df[
        (df.AGEP >= 25)
        & (df.AGEP <= 64)
        & (df.WAGP > 0)
        & (df.ESR.isin([1, 2]))
        & (df.COW.isin([1, 2, 3, 4, 5]))
    ].copy()

    df["Yresp"] = np.log(df.WAGP.values + 1.0)
    df["age"] = df.AGEP.values
    df["educ"] = df.SCHL.values
    df["hrs"] = df.WKHP.values
    df["male"] = (df.SEX.values == 1).astype(float)
    df["hisp"] = (df.HISP.values > 1).astype(float)
    df["white"] = (df.RAC1P.values == 1).astype(float)
    df["public"] = df.COW.isin([3, 4, 5]).astype(float)

    continuous = ["age", "educ", "hrs"]
    binary = ["male", "hisp", "white", "public"]

    df = df[df[continuous].notna().all(axis=1)].copy()

    states = sorted(df.ST.unique())
    states = sorted(
        states,
        key=lambda s: (FIPS.get(int(s), str(int(s))) != central_state,
                       FIPS.get(int(s), str(int(s)))),
    )

    train_blocks = []
    test_blocks = []

    for s in states:
        sub = df[df.ST == s]
        state_name = FIPS.get(int(s), str(int(s)))

        idx = rng.permutation(len(sub))
        n_train = min(n_per_state, len(sub))

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        values = sub[continuous + binary + ["Yresp"]].values

        Xtr_c = values[train_idx][:, :3]
        Xte_c = values[test_idx][:, :3]

        mu = Xtr_c.mean(axis=0)
        sd = Xtr_c.std(axis=0)
        sd[sd == 0] = 1.0

        Xtr = np.column_stack([(Xtr_c - mu) / sd, values[train_idx][:, 3:7]])
        Xte = np.column_stack([(Xte_c - mu) / sd, values[test_idx][:, 3:7]])

        Xtr[:, :3] = np.clip(Xtr[:, :3], -winsor_limit, winsor_limit)
        Xte[:, :3] = np.clip(Xte[:, :3], -winsor_limit, winsor_limit)

        train_blocks.append((state_name, Xtr, values[train_idx][:, 7]))
        test_blocks.append((state_name, Xte, values[test_idx][:, 7]))

    covariate_names = [
        "Age",
        "Education",
        "Hours/week",
        "Sex (male=1)",
        "Hispanic (yes=1)",
        "Race (White=1)",
        "Public sector (yes=1)",
    ]

    return train_blocks, test_blocks, covariate_names
