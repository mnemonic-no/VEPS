from typing import Any

from veps.config import DistilBertHyperparameters, fetch_config_from_yaml
from veps.distilbert.cvss import CVSSTrainer
from veps.distilbert.cwe import CWETrainer
from veps.paths import TRAINING_DIR, MODELS_DIR


def _resolve_config(args: Any) -> DistilBertHyperparameters:
    """Resolve the effective DistilBert config (CLI > YAML > defaults).

    Tries to load ``config.yaml``; if that fails (file missing, malformed,
    or missing required XGBoost fields), falls back to a plain
    ``DistilBertHyperparameters()`` so ``train-bert`` is usable without a
    full pipeline config in place.
    """
    try:
        cfg = fetch_config_from_yaml()
        base = cfg.distilbert
    except Exception as e:
        print(f"[train-bert] note: could not load config.yaml ({e}); using built-in defaults.")
        base = DistilBertHyperparameters()

    # CLI overrides — only apply fields the user actually set (default=None).
    overrides: dict[str, Any] = {}
    if getattr(args, "epochs", None) is not None:
        overrides["num_epochs"] = args.epochs
    if getattr(args, "batch_size_cvss", None) is not None:
        overrides["batch_size_cvss"] = args.batch_size_cvss
    if getattr(args, "batch_size_cwe", None) is not None:
        overrides["batch_size_cwe"] = args.batch_size_cwe
    if getattr(args, "lr", None) is not None:
        overrides["learning_rate"] = args.lr
    if getattr(args, "max_length", None) is not None:
        overrides["max_length"] = args.max_length
    if getattr(args, "patience", None) is not None:
        overrides["patience"] = args.patience
    if getattr(args, "seeds", None) is not None:
        overrides["seeds"] = list(args.seeds)
    if getattr(args, "test_months", None) is not None:
        overrides["test_months"] = args.test_months
    if getattr(args, "val_months", None) is not None:
        overrides["val_months"] = args.val_months
    if getattr(args, "num_workers", None) is not None:
        overrides["num_workers"] = args.num_workers
    if getattr(args, "no_amp", False):
        overrides["amp_enabled"] = False

    if overrides:
        return base.model_copy(update=overrides)
    return base


def _print_resolved_config(config: DistilBertHyperparameters) -> None:
    print("[train-bert] resolved DistilBert config:")
    for k, v in config.model_dump().items():
        print(f"[train-bert]   {k}: {v}")


def train_cvss(config: DistilBertHyperparameters):
    """Train the CVSS model."""
    print("Starting CVSS model training...")
    trainer = CVSSTrainer(
        data_dir=TRAINING_DIR / 'cvss',
        models_dir=MODELS_DIR / 'distilbert',
        config=config,
    )

    model = trainer.train()
    print("CVSS model training completed!")
    return model


def train_cwe(config: DistilBertHyperparameters):
    """Train the CWE model."""
    print("Starting CWE model training...")
    trainer = CWETrainer(
        data_dir=TRAINING_DIR / 'cwe',
        models_dir=MODELS_DIR / 'distilbert',
        config=config,
    )

    model = trainer.train()
    print("CWE model training completed!")
    return model


def main(args):
    config = _resolve_config(args)
    _print_resolved_config(config)

    if getattr(args, "dry_run", False):
        print("[train-bert] --dry-run set; exiting before training.")
        return

    if args.model == 'cvss':
        _ = train_cvss(config)
    elif args.model == 'cwe':
        _ = train_cwe(config)
    elif args.model == 'all':
        _ = train_cvss(config)
        _ = train_cwe(config)
        print("All model training completed!")
