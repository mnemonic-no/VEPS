from pathlib import Path
import argparse



from veps.lib.data.preprocessing import create_training_set, create_inference_dataset
from veps.config import DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="Create training set from extracted features")
    parser.add_argument("--features-file", type=Path, help="Path to extracted features CSV")
    parser.add_argument("--output-file", type=Path, help="Output training set file")
    parser.add_argument("--inference-only", action="store_true", help="Create inference dataset only")
    parser.add_argument("--window-size", type=int, default=30, help="Window size in days")
    parser.add_argument("--prediction-horizon", type=int, default=30, help="Prediction horizon in days")
    parser.add_argument("--stride", type=int, default=30, help="Stride between windows in days")
    parser.add_argument("--sampling-strategy", default="severity_weighted", 
                       choices=["balanced", "severity_weighted", "temporal_matched"],
                       help="Negative sampling strategy")
    
    args = parser.parse_args()
    
    if args.inference_only:
        if not args.features_file:
            features_files = list((DATA_DIR / "preprocessed").glob("nvd_features_*.csv"))
            if not features_files:
                print("No features files found. Run feature extraction first.")
                return
            args.features_file = max(features_files, key=lambda p: p.stat().st_mtime)
        
        output_file = create_inference_dataset(args.features_file)
        print(f"Inference dataset created: {output_file}")
    else:
        output_file = create_training_set(
            features_file=args.features_file,
            output_file=args.output_file,
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon,
            stride=args.stride,
            sampling_strategy=args.sampling_strategy,
        )
        print(f"Training set created: {output_file}")

if __name__ == "__main__":
    main()