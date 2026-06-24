import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cst_dcqr.config import load_beijing_config, resolve_path
from cst_dcqr.beijing_data import load_beijing_blocks
from cst_dcqr.analysis import run_cst_dcqr_analysis
from cst_dcqr.metrics import (
    build_intercept_table,
    build_check_loss_table,
    build_slope_table_for_beijing,
    build_distance_table,
)
from cst_dcqr.io_utils import ensure_dir, save_csv, save_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Run Beijing CST-DCQR analysis.")
    parser.add_argument("--config", default="configs/beijing_config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_beijing_config(REPO_ROOT / args.config)

    data_dir = resolve_path(cfg.data_dir, REPO_ROOT)
    output_dir = ensure_dir(resolve_path(cfg.output_dir, REPO_ROOT))

    train_blocks, test_blocks, covariate_names = load_beijing_blocks(
        data_dir=data_dir,
        covariates=cfg.covariates,
        split_date=cfg.split_date,
        central_station=cfg.central_station,
        winsor_limit=cfg.winsor_limit,
    )

    results = run_cst_dcqr_analysis(
        train_blocks=train_blocks,
        test_blocks=test_blocks,
        M=cfg.M,
        eps_list=cfg.eps_list,
        seed=cfg.seed,
    )

    save_npz(
        output_dir / "beijing_results.npz",
        taus=results["taus"],
        alpha_gscqr=results["alpha_gscqr"],
        alpha_csl=results["alpha_csl"],
        alpha_cst=results["alpha_cst"],
        alpha_dp05=results["alpha_dp"][0.5],
        alpha_dp1=results["alpha_dp"][1.0],
        alpha_dp5=results["alpha_dp"][5.0],
        beta_g=results["beta_g"],
        lo_g=results["lo_g"],
        hi_g=results["hi_g"],
        beta_c=results["beta_c"],
        lo_c=results["lo_c"],
        hi_c=results["hi_c"],
        beta_csl=results["beta_csl"],
        beta_dp05=results["beta_dp"][0.5],
        beta_dp1=results["beta_dp"][1.0],
        beta_dp5=results["beta_dp"][5.0],
        dist_cst=results["dist_cst"],
        dist_csl=results["dist_csl"],
        n_sizes=results["n_sizes"],
        N_train=results["N_train"],
        N_test=results["N_test"],
        n1=results["n1"],
    )

    intercept_df = build_intercept_table(
        results["method_alpha"], results["taus"], cfg.M, response_label="pm25"
    )
    loss_df = build_check_loss_table(
        results["method_alpha"], test_blocks, results["taus"], cfg.M
    )
    distance_df = build_distance_table(results)

    save_csv(intercept_df, output_dir / "beijing_intercepts.csv")
    save_csv(loss_df, output_dir / "beijing_check_loss.csv")
    save_csv(distance_df, output_dir / "beijing_method_distances.csv")

    if cfg.export_slope_coefficients:
        slope_df = build_slope_table_for_beijing(covariate_names, cfg.M, results)
        save_csv(slope_df, output_dir / "beijing_slope_coefficients.csv")


if __name__ == "__main__":
    main()
