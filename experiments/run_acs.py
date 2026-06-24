import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cst_dcqr.config import load_acs_config, resolve_path
from cst_dcqr.acs_data import load_acs_blocks
from cst_dcqr.analysis import run_cst_dcqr_analysis
from cst_dcqr.metrics import (
    build_intercept_table,
    build_check_loss_table,
    build_distance_table,
)
from cst_dcqr.io_utils import ensure_dir, save_csv, save_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACS CST-DCQR analysis.")
    parser.add_argument("--config", default="configs/acs_config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_acs_config(REPO_ROOT / args.config)

    data_path = resolve_path(cfg.data_path, REPO_ROOT)
    output_dir = ensure_dir(resolve_path(cfg.output_dir, REPO_ROOT))

    train_blocks, test_blocks, covariate_names = load_acs_blocks(
        data_path=data_path,
        n_per_state=cfg.n_per_state,
        central_state=cfg.central_state,
        seed=cfg.seed,
        winsor_limit=cfg.winsor_limit,
    )

    results = run_cst_dcqr_analysis(
        train_blocks=train_blocks,
        test_blocks=test_blocks,
        M=cfg.M,
        eps_list=cfg.eps_list,
        seed=cfg.seed,
    )

    # The full numerical results are saved for reproducibility.
    # No ACS slope-coefficient table or LaTeX table-generation code is included.
    save_npz(
        output_dir / "acs_results.npz",
        taus=results["taus"],
        alpha_gscqr=results["alpha_gscqr"],
        alpha_csl=results["alpha_csl"],
        alpha_cst=results["alpha_cst"],
        alpha_dp05=results["alpha_dp"][0.5],
        alpha_dp1=results["alpha_dp"][1.0],
        alpha_dp5=results["alpha_dp"][5.0],
        dist_cst=results["dist_cst"],
        dist_csl=results["dist_csl"],
        n_sizes=results["n_sizes"],
        N_train=results["N_train"],
        N_test=results["N_test"],
        n1=results["n1"],
    )

    intercept_df = build_intercept_table(
        results["method_alpha"], results["taus"], cfg.M, response_label="wage"
    )
    loss_df = build_check_loss_table(
        results["method_alpha"], test_blocks, results["taus"], cfg.M
    )
    distance_df = build_distance_table(results)

    save_csv(intercept_df, output_dir / "acs_intercepts.csv")
    save_csv(loss_df, output_dir / "acs_check_loss.csv")
    save_csv(distance_df, output_dir / "acs_method_distances.csv")

    if cfg.export_slope_coefficients:
        raise RuntimeError(
            "ACS slope-coefficient table export is disabled by design. "
            "This repository intentionally omits the code for the ACS common-slope table."
        )


if __name__ == "__main__":
    main()
