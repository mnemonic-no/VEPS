from pathlib import Path

# Data specific
DATA_DIR = Path(__file__).parent.parent.parent / "data"
NVD_FILEPATH: Path = Path(__file__).parent.parent.parent.resolve() / "data" / "raw" / "nvd"
# Training data for CWE/CVSS classifiers
TRAINING_DIR = DATA_DIR / "training_dir"
# NVD data with predictions added
NVD_ENRICHED = DATA_DIR / "nvd_data_with_predictions"
# Path for feature extraction script
PREPROCESS_FILEPATH = DATA_DIR / "preprocessed"
# Observed CVE, for every day
OBSERVATIONS_PATH = DATA_DIR / "cve_observations.csv"
# CVE mentions in threat reports
CVE_MENTIONS_PATH = DATA_DIR / "cve_mentions.json"
# NVD data feeds with predicted values
CORPUS_FILEPATH = DATA_DIR / "nvd_data_with_predictions"

VEPS_TRAINING_SETS = DATA_DIR / "veps_training_sets"

INFERENCE_DATASETS = DATA_DIR / "inference_datasets"

MODELS_DIR = DATA_DIR / "models"

DAILY_PREDICTIONS = DATA_DIR / "predictions" / "daily"


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        NVD_FILEPATH,
        TRAINING_DIR,
        NVD_ENRICHED,
        PREPROCESS_FILEPATH,
        MODELS_DIR,
        CORPUS_FILEPATH,
        VEPS_TRAINING_SETS,
        INFERENCE_DATASETS,
        MODELS_DIR,
        DAILY_PREDICTIONS
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("All necessary directories created/verified")