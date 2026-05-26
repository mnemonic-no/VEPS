"""VEPS command-line interface.

Single entry point with subparsers for every pipeline stage. Shared root flags
(`--config`, `--log-level`) apply before the subcommand name. Each subcommand
dispatches to a `main(args)` function in the corresponding stage module.

Subcommand handlers are imported lazily so that lightweight stages (e.g.
``veps download``) do not pull in heavy optional dependencies such as torch.
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

from veps.paths import NVD_FILEPATH


HANDLERS = {
    "download":         ("veps.data.download_nvd",               "main"),
    "extract-features": ("veps.data.feature_extraction",         "main"),
    "build-trainset":   ("veps.data.exploit_training_builder",    "main"),
    "train":            ("veps.exploit_model.train",             "main"),
    "predict":          ("veps.exploit_model.predict",           "main"),
    "tune":             ("veps.exploit_model.parameter_tuning",  "main"),
    "train-bert":       ("veps.distilbert.train",                "main"),
    "predict-bert":     ("veps.data.nvd_enrichment",             "main"),
    "extract-bert-trainset": ("veps.data.distilbert_corpus_builder",   "main"),
    "daily":            ("veps.daily",                           "main"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veps",
        description="VEPS — Vulnerability Exploitation Prediction Score CLI",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("download", description="Download NVD CVE data")
    p.add_argument("--years", type=str, help="Year range (e.g., '2020-2024')")
    p.add_argument("--latest", action="store_true", help="Download latest data only")
    p.add_argument("--output-dir", type=str, default=NVD_FILEPATH, help="Output directory")

    p = subparsers.add_parser("extract-features", description="Extract features from NVD CVE data")
    p.add_argument("--input-dir", type=Path, help="Input directory with NVD JSON files")
    p.add_argument("--output-file", type=Path, help="Output CSV file path")
    p.add_argument("--latest-only", action="store_true", help="Process only the most recent files")

    p = subparsers.add_parser("build-trainset", description="Create training set from extracted features")
    p.add_argument("--features-file", type=Path, help="Path to extracted features CSV")
    p.add_argument("--output-file", type=Path, help="Output training set file")
    p.add_argument("--inference-only", action="store_true", help="Create inference dataset only")
    # default=None so we can distinguish "user passed nothing" (use Config)
    # from "user passed the same value as Config".
    p.add_argument("--window-size", type=int, default=None, help="Window size in days (default: Config value)")
    p.add_argument("--prediction-horizon", type=int, default=None, help="Prediction horizon in days (default: Config value)")
    p.add_argument("--stride", type=int, default=None, help="Stride between windows in days (default: Config value)")

    p = subparsers.add_parser("train", description="Train vulnerability prediction model")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="XGBoost compute device (overrides config)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory to write the trained pipeline, trained "
                        "categories, and (when --params is omitted) read "
                        "best_params.json from "
                        "(default: data/models/vuln_pred). Use distinct "
                        "directories to train side-by-side models.")
    p.add_argument("--params", type=Path, default=None,
                   help="Path to tuned XGBoost params JSON "
                        "(default: <model-dir>/best_params.json, "
                        "written by `veps tune`)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip post-fit isotonic/sigmoid calibration "
                        "(overrides config.calibration.enabled). Useful for "
                        "ablation runs comparing raw vs calibrated.")
    p.add_argument("--deploy", action="store_true",
                   help="Deploy mode: train on all data minus the tail "
                        "calibration slice; skip the holdout split and its "
                        "metric blocks. Use only after a diagnostic run has "
                        "validated calibration. Ignores config.cutoff_date.")

    p = subparsers.add_parser("predict", description="Make daily vulnerability predictions")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory containing the trained pipeline + "
                        "trained_categories.json "
                        "(default: data/models/vuln_pred). When set to a "
                        "non-default dir, the output filename is suffixed "
                        "with the directory name so parallel runs do not "
                        "clobber each other.")

    p = subparsers.add_parser("tune", description="Tune hyperparameters for vulnerability prediction")
    p.add_argument("--training-file", type=Path, help="Path to training data file")
    p.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    p.add_argument("--tune-only", action="store_true", help="Only tune, don't train final model")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="XGBoost compute device (overrides config)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory to write best_params.json (and, unless "
                        "--tune-only, the trained pipeline + categories) "
                        "(default: data/models/vuln_pred). Use distinct "
                        "directories to tune side-by-side models.")
    p.add_argument("--study-name", type=str, default=None,
                   help="Optuna study name (resumes if it already exists in storage; "
                        "default: config.tune.study_name)")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL (default: config.tune.storage)")

    p = subparsers.add_parser("train-bert", description="Train DistilBERT models for vulnerability analysis")
    p.add_argument(
        "--model",
        choices=["cvss", "cwe", "all"],
        default="all",
        help="Which model(s) to train: cvss, cwe, or all (default: all)",
    )
    # Hyperparameter overrides. `default=None` so the trainer can tell
    # "user passed nothing" from "user passed the same value as the
    # config default" — only non-None values override the resolved config.
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--batch-size-cvss", type=int, default=None, help="Batch size for CVSS trainer")
    p.add_argument("--batch-size-cwe", type=int, default=None, help="Batch size for CWE trainer")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--max-length", type=int, default=None, help="Tokenizer max sequence length")
    p.add_argument("--patience", type=int, default=None, help="Early-stopping patience (epochs)")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more integer seeds to train across",
    )
    p.add_argument("--test-months", type=int, default=None, help="Months at the end held out as test")
    p.add_argument("--val-months", type=int, default=None, help="Months immediately before test held out as val")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count")
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (autocast). Off by default; gated per-device.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the effective config, then exit without training.",
    )

    p = subparsers.add_parser("predict-bert", description="Add ML predictions to NVD data")
    p.add_argument("--input-dir", type=Path, help="Input directory with NVD JSON files")
    p.add_argument("--output-dir", type=Path, help="Output directory for processed files")
    p.add_argument("--single-file", type=Path, help="Process a single file")
    p.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="SQLite prediction cache path (default: data/interim/prediction_cache.sqlite)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the prediction cache and run BERT on every CVE",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Drop all cached predictions before processing (full re-enrichment)",
    )

    p = subparsers.add_parser(
        "extract-bert-trainset",
        description="Extract training data for DistilBERT CWE/CVSS classifiers from the NVD corpus",
    )
    p.add_argument(
        "--cwe-keep-top-n",
        type=int,
        default=None,
        help="Keep only the N most frequent CWEs (default: config.distilbert.cwe_keep_top_n)",
    )
    p.add_argument("--show-cwe-distribution", action="store_true", help="Print the top CWE distribution after extraction")

    p = subparsers.add_parser("daily", description="Run daily pipelines")
    p.add_argument("--download-nvd", action="store_true", help="Download the current year's NVD feed before processing")
    p.add_argument("--predict-cvss", action="store_true", help="Run DistilBERT CVSS+CWE prediction before feature extraction")
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Drop all cached BERT predictions before --predict-cvss (full re-enrichment)",
    )
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory containing the trained pipeline + "
                        "trained_categories.json "
                        "(default: data/models/vuln_pred). When set to a "
                        "non-default dir, the output filename is suffixed "
                        "with the directory name so parallel runs do not "
                        "clobber each other.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    if args.config is not None:
        # Rebind on the module object so downstream callers that access
        # paths.CONFIG_FILE_PATH see the override. Importing the name into a
        # caller's namespace (`from .paths import CONFIG_FILE_PATH`) would
        # capture the original value and ignore this assignment.
        import veps.paths
        veps.paths.CONFIG_FILE_PATH = args.config.resolve()

    mod_name, fn_name = HANDLERS[args.command]
    mod = importlib.import_module(mod_name)
    result = getattr(mod, fn_name)(args)
    if isinstance(result, int):
        sys.exit(result)
