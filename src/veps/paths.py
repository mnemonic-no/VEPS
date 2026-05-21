import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if not (PROJECT_ROOT / "pyproject.toml").exists():
    warnings.warn(
        f"PROJECT_ROOT ({PROJECT_ROOT}) is not a VEPS repo root; "
        "data paths may be incorrect. Run from the repo or set up "
        "a project directory.",
        RuntimeWarning,
        stacklevel=2,
    )

DATA_DIR = PROJECT_ROOT / "data"
NVD_FILEPATH = (PROJECT_ROOT / "data" / "raw" / "nvd").resolve()
OBSERVATIONS_PATH = DATA_DIR / "raw" / "cve_observations.csv"
CVE_MENTIONS_PATH = DATA_DIR / "raw" / "cve_mentions.json"
PREPROCESS_FILEPATH = DATA_DIR / "interim"
NVD_ENRICHED = DATA_DIR / "interim" / "nvd_with_predictions"
CORPUS_FILEPATH = DATA_DIR / "interim" / "nvd_with_predictions"
PREDICTION_CACHE = DATA_DIR / "interim" / "prediction_cache.sqlite"
TRAINING_DIR = DATA_DIR / "processed" / "training"
VEPS_TRAINING_SETS = DATA_DIR / "processed"
INFERENCE_DATASETS = DATA_DIR / "inference"
MODELS_DIR = DATA_DIR / "models"
DAILY_PREDICTIONS = DATA_DIR / "predictions" / "daily"

TRAINED_MODEL_DIR = MODELS_DIR / "vuln_pred"
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# DistilBERT artifact paths
#
# All seed-suffixed checkpoint and threshold paths are constructed here so
# trainer/classifier code never concatenates filenames inline.
# ---------------------------------------------------------------------------

def cvss_checkpoint_path(models_dir: Path, seed: int) -> Path:
    return models_dir / f"cvss_seed{seed}_best.pth"


def cwe_checkpoint_path(models_dir: Path, seed: int) -> Path:
    return models_dir / f"cwe_seed{seed}_best.pth"


def cwe_thresholds_path(models_dir: Path, seed: int) -> Path:
    return models_dir / f"cwe_thresholds_seed{seed}.json"


def cvss_run_manifest_path(models_dir: Path) -> Path:
    return models_dir / "cvss_run.json"


def cwe_run_manifest_path(models_dir: Path) -> Path:
    return models_dir / "cwe_run.json"


def cvss_token_lengths_path(models_dir: Path) -> Path:
    return models_dir / "cvss_token_lengths.json"


def cwe_token_lengths_path(models_dir: Path) -> Path:
    return models_dir / "cwe_token_lengths.json"
