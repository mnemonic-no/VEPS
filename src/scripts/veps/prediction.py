
import sys
from pathlib import Path
from datetime import datetime

from veps.lib.models.veps.predict import predict_vulnerabilities
from veps.config import DATA_DIR, MODELS_DIR

def main():
    inference_files = list((DATA_DIR / "preprocessed").glob("inference_dataset_*.csv"))
    if not inference_files:
        sys.exit(1)
    
    inference_file = max(inference_files, key=lambda p: p.stat().st_mtime)
    
    # Create dated output file
    predictions_dir = DATA_DIR / "predictions" / "daily"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = predictions_dir / f"predictions_{date_str}.csv"
    
    predict_vulnerabilities(
        inference_file, 
        model_dir=MODELS_DIR / "vuln_pred",
        output_file=output_file
    )

if __name__ == "__main__":
    main()