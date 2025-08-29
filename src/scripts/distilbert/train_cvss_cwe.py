import argparse
from pathlib import Path

from veps.lib.models.distilbert.cvss.trainer import CVSSTrainer
from veps.lib.models.distilbert.cwe.trainer import CWETrainer
from veps.config import TRAINING_DIR, MODELS_DIR


def train_cvss():
    """Train the CVSS model."""
    print("Starting CVSS model training...")
    trainer = CVSSTrainer(
        data_dir=TRAINING_DIR / 'cvss',
        models_dir=MODELS_DIR / 'distilbert'
    )
    
    model = trainer.train()
    print("CVSS model training completed!")
    return model


def train_cwe():
    """Train the CWE model."""
    print("Starting CWE model training...")
    trainer = CWETrainer(
        data_dir=TRAINING_DIR / 'cwe',
        models_dir=MODELS_DIR / 'distilbert'
    )
    
    model = trainer.train()
    print("CWE model training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT models for vulnerability analysis')
    parser.add_argument(
        '--model', 
        choices=['cvss', 'cwe', 'all'], 
        default='all',
        help='Which model(s) to train: cvss, cwe, or all (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'cvss':
        _ = train_cvss()
    elif args.model == 'cwe':
        _ = train_cwe()
    elif args.model == 'all':
        _ = train_cvss()
        _ = train_cwe()
        print("All model training completed!")


if __name__ == "__main__":
    main()