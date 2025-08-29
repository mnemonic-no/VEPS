

import argparse
import logging
from pathlib import Path
from datetime import datetime

from veps.lib.data.download_nvd import download_nvd_data_range
from veps.lib.data.feature_extraction import extract_features_from_directory
from veps.lib.data.preprocessing import create_inference_dataset
from veps.lib.data.nvd_processor import NVDProcessor
from veps.lib.models.veps.predict import predict_vulnerabilities
from veps.config import MODELS_DIR, CORPUS_FILEPATH, PREPROCESS_FILEPATH, NVD_FILEPATH, DAILY_PREDICTIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily pipelines")
    parser.add_argument("--download-nvd", action="store_true", help="Create inference dataset only")
    parser.add_argument("--predict-cvss", action="store_true", help="Create inference dataset only")

    return parser.parse_args()

def run_daily_pipeline():
    args = parse_args()
    if args.download_nvd:
        current_year = datetime.now().year
        _ = download_nvd_data_range(NVD_FILEPATH, end_year=current_year)
    
    # Optionaly add predicted values for CVSS/CWE
    if args.predict_cvss:
        models_dir = MODELS_DIR / 'distilbert'
        processor = NVDProcessor(models_dir)
        processed_files = processor.process_directory(NVD_FILEPATH, CORPUS_FILEPATH)
        print(f"Successfully processed {len(processed_files)} files")
    
    # Extract features
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PREPROCESS_FILEPATH / f"nvd_features_{timestamp}.csv"
    feature_file = extract_features_from_directory(CORPUS_FILEPATH, output_file)
    
    # Create inference dataset
    inference_data = create_inference_dataset(feature_file)
    
    # Make predictions
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = DAILY_PREDICTIONS / f"predictions_{date_str}.csv"
    
    _ = predict_vulnerabilities(
        inference_data, 
        MODELS_DIR / "vuln_pred",
        output_file
    )

if __name__ == "__main__":
    run_daily_pipeline()