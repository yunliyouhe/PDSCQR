# CST-DCQR / DP-CST-DCQR Real-Data Code

This repository contains a clean, modular version of the real-data analysis code
for CST-DCQR and DP-CST-DCQR.

The repository is designed for GitHub reproducibility:

- no personal absolute paths.

The code only runs the real-data analyses and exports numerical results.
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

