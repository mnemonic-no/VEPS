#!/usr/bin/env python3
"""Extract features from NVD JSON files."""

import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from veps.lib.data.feature_extraction import (
    extract_features_from_directory,
    extract_latest_features
)
from veps.config import CORPUS_FILEPATH, PREPROCESS_FILEPATH

def main():
    parser = argparse.ArgumentParser(description="Extract features from NVD CVE data")
    parser.add_argument("--input-dir", type=Path, help="Input directory with NVD JSON files")
    parser.add_argument("--output-file", type=Path, help="Output CSV file path")
    parser.add_argument("--latest-only", action="store_true", help="Process only the most recent files")
    
    args = parser.parse_args()
    
    if args.input_dir and args.output_file:
        # Custom input/output
        output_file = extract_features_from_directory(args.input_dir, args.output_file)
    elif args.latest_only:
        # Process only the most recent NVD files
        output_file = extract_latest_features()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PREPROCESS_FILEPATH / f"nvd_features_{timestamp}.csv"
        output_file = extract_features_from_directory(CORPUS_FILEPATH, output_file)
    
    print(f"Feature extraction completed: {output_file}")

if __name__ == "__main__":
    main()