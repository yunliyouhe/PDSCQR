# CST-DCQR / DP-CST-DCQR Real-Data Code

This repository contains a clean, modular version of the real-data analysis code
for CST-DCQR and DP-CST-DCQR.

The repository is designed for GitHub reproducibility:

- no LaTeX table-generation code,
- no plotting code,
- no personal absolute paths,
- no code for the ACS slope-coefficient table, namely the table captioned
  "Estimated common slope coefficients ... for the ACS wage income application ...".

The code only runs the real-data analyses and exports numerical results.

## Project structure

```text
CST-DCQR-GitHub-NoTable11/
├── README.md
├── requirements.txt
├── configs/
│   ├── beijing_config.json
│   └── acs_config.json
├── data/
│   └── README.md
├── experiments/
│   ├── run_beijing.py
│   └── run_acs.py
├── results/
│   └── .gitkeep
└── src/
    └── cst_dcqr/
        ├── __init__.py
        ├── acs_data.py
        ├── analysis.py
        ├── beijing_data.py
        ├── config.py
        ├── covariance.py
        ├── estimators.py
        ├── io_utils.py
        ├── kernels.py
        ├── metrics.py
        └── privacy.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

### Beijing

Download the Beijing Multi-Site Air-Quality dataset and put the CSV files in:

```text
data/PRSA_Data_20130301-20170228/
```

or change `data_dir` in `configs/beijing_config.json`.

### ACS

Put the prepared ACS subset pickle file in:

```text
data/acs_subset.pkl
```

or change `data_path` in `configs/acs_config.json`.

## Run analyses

```bash
python experiments/run_beijing.py --config configs/beijing_config.json
python experiments/run_acs.py --config configs/acs_config.json
```

## Output

Beijing output files:

```text
results/beijing/beijing_results.npz
results/beijing/beijing_intercepts.csv
results/beijing/beijing_check_loss.csv
results/beijing/beijing_slope_coefficients.csv
results/beijing/beijing_method_distances.csv
```

ACS output files:

```text
results/acs/acs_results.npz
results/acs/acs_intercepts.csv
results/acs/acs_check_loss.csv
results/acs/acs_method_distances.csv
```

The ACS common-slope table export is intentionally omitted.
