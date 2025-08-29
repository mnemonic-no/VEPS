#!/usr/bin/env python3
"""Hyperparameter tuning for vulnerability prediction model."""

from pathlib import Path
import argparse


from veps.lib.models.veps.parameter_tuning import tune_hyperparameters, train_with_best_params
from veps.lib.models.veps.data_manager import save_pipeline
from veps.config import DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for vulnerability prediction")
    parser.add_argument("--training-file", type=Path, help="Path to training data file")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--tune-only", action="store_true", help="Only tune, don't train final model")
    
    args = parser.parse_args()
    
    # Find training file if not specified
    if args.training_file is None:
        training_files = list((DATA_DIR).glob("training_set_*.csv"))
        if not training_files:
            print("No training set found. Run create_training_set.py first.")
            return
        args.training_file = max(training_files, key=lambda p: p.stat().st_mtime)
        print(f"Using training file: {args.training_file}")
    
    # Tune hyperparameters
    print("Starting hyperparameter tuning...")
    best_params = tune_hyperparameters(args.training_file, args.n_trials)
    
    if not args.tune_only:
        # Train final model with best parameters
        print("Training final model with best parameters...")
        pipeline = train_with_best_params(args.training_file, best_params)
        
        # Save the tuned model
        save_pipeline(pipeline)
        print("Tuned model saved!")
    
    print(f"\nBest hyperparameters: {best_params}")

if __name__ == "__main__":
    main()