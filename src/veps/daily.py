from pathlib import Path
from datetime import datetime

from veps.config import fetch_config_from_yaml
from veps.data.download_nvd import download_nvd_data_range
from veps.data.feature_extraction import extract_features_from_directory
from veps.data.exploit_training_builder import create_inference_dataset
from veps.data.nvd_enrichment import NVDEnricher
from veps.data.prediction_cache import PredictionCache
from veps.exploit_model.predict import predict_vulnerabilities
from veps.paths import (
    CORPUS_FILEPATH,
    DAILY_PREDICTIONS,
    MODELS_DIR,
    NVD_FILEPATH,
    PREDICTION_CACHE,
    PREPROCESS_FILEPATH,
)


def main(args):
    cfg = fetch_config_from_yaml()

    if args.download_nvd:
        current_year = datetime.now().year
        _ = download_nvd_data_range(NVD_FILEPATH, end_year=current_year)

    # Optionaly add predicted values for CVSS/CWE
    if args.predict_cvss:
        models_dir = MODELS_DIR / 'distilbert'
        with PredictionCache(PREDICTION_CACHE) as cache:
            if getattr(args, "rebuild_cache", False):
                print(f"Rebuilding prediction cache at {PREDICTION_CACHE}")
                cache.clear()
            else:
                print(
                    f"Using prediction cache at {PREDICTION_CACHE} "
                    f"(entries={cache.size()})"
                )
            enricher = NVDEnricher(models_dir, cache=cache)
            processed_files = enricher.process_directory(NVD_FILEPATH, CORPUS_FILEPATH)
            print(f"Successfully processed {len(processed_files)} files")

    # Extract features
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PREPROCESS_FILEPATH / f"nvd_features_{timestamp}.csv"
    feature_file = extract_features_from_directory(CORPUS_FILEPATH, output_file)

    # Create inference dataset
    inference_data = create_inference_dataset(
        feature_file, window_size=cfg.preprocessing.window_size
    )

    # Make predictions
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = DAILY_PREDICTIONS / f"predictions_{date_str}.csv"

    _ = predict_vulnerabilities(
        inference_data,
        MODELS_DIR / "vuln_pred",
        output_file
    )
