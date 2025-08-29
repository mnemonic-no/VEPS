#!/usr/bin/env python3
"""Train vulnerability prediction model."""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

from veps.lib.models.veps.data_manager import load_dataset, split_data_by_date, save_pipeline
from veps.lib.models.veps.pipeline import create_full_pipeline
from veps.lib.models.veps.config import fetch_config_from_yaml
from veps.config import DATA_DIR

def train_model():
    """Train the vulnerability prediction model."""
    config = fetch_config_from_yaml()
    
    # Find latest training data
    training_files = list((DATA_DIR).glob("training_set_*.csv"))
    if not training_files:
        print("No training set found. Run create_training_set.py first.")
        return
    
    training_file = max(training_files, key=lambda p: p.stat().st_mtime)
    print(f"Using training file: {training_file}")
    
    df = load_dataset(training_file)
    
    # Split data based on cutoff date, 
    if config.get('cutoff_date'):
        cutoff_date = pd.to_datetime(config['cutoff_date']).tz_localize('UTC')
        train_df, test_df = split_data_by_date(df, cutoff_date)
    else:
        cutoff_date = df['window_end'].quantile(0.8)
        train_df, test_df = split_data_by_date(df, cutoff_date)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    target_col = config['target']
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    classifier = XGBClassifier(**config['hyperparameters'])
    
    pipeline = create_full_pipeline(classifier, config['feature_columns'])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\nModel Performance:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    save_pipeline(pipeline)
    print("Model training complete!")

if __name__ == "__main__":
    train_model()