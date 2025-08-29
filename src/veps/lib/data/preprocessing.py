import json
from pathlib import Path
from typing import Tuple, Dict
import warnings

import pandas as pd
import numpy as np
from sklearn.utils import resample

from ...config import OBSERVATIONS_PATH, CVE_MENTIONS_PATH, DATA_DIR, INFERENCE_DATASETS


def load_cve_mentions(input_file: Path) -> Dict:
    """Load CVE mentions from JSON file."""
    with open(input_file, 'r') as f:
        cve_mentions = json.load(f)
    
    # Convert string dates back to timezone-naive pandas Timestamp objects
    for cve in cve_mentions:
        cve_mentions[cve] = {pd.Timestamp(date): count 
                             for date, count in cve_mentions[cve].items()}
    
    return cve_mentions


def load_and_preprocess_data(
    observations_path: Path, features_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess observation data and extracted features.
    
    Args:
        observations_path: Path to observations CSV file
        features_path: Path to extracted features CSV file (from feature_extraction.py)
    
    Returns:
        Tuple of (observations_df, features_df)
    """
    try:
        observations = pd.read_csv(observations_path)
        features = pd.read_csv(features_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to load data files: {e}")

    # Process observations data
    observations["date"] = pd.to_datetime(observations["date"], utc=True)
    
    # Process features data (from feature_extraction.py output)
    features["published_date"] = pd.to_datetime(features["published_date"], utc=True)
    features["last_modified_date"] = pd.to_datetime(features["last_modified_date"], utc=True)

    # Merge observations with features
    df = pd.merge(
        observations, features, left_on="cve", right_on="cve_id", how="left"
    )
    df = df.sort_values("date")
    features = features.sort_values("published_date")

    return df, features


def negative_sampling(
    window_cves: set,
    all_cves: set,
    features: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    n_negative_per_positive: int = 2,
    sampling_strategy: str = "balanced"
) -> list:
    """
    Negative sampling using available data features.
    
    Args:
        window_cves: Set of CVEs with exploits in the current window
        all_cves: Set of all available CVEs
        features: DataFrame with CVE features
        window_start: Start of the current window
        window_end: End of the current window
        n_negative_per_positive: Number of negative samples per positive
        sampling_strategy: Strategy for sampling ("balanced", "severity_weighted", "temporal_matched")
    
    Returns:
        List of negative CVE IDs
    """
    # Get potential negatives: CVEs published before window end but not exploited in window
    potential_negatives = all_cves - window_cves
    
    # Filter to CVEs that were "available" (published before window end)
    available_negatives = features[
        (features["cve_id"].isin(potential_negatives)) &
        (features["published_date"] < window_end)
    ].copy()
    
    if len(available_negatives) == 0:
        return []
    
    n_samples = min(
        len(window_cves) * n_negative_per_positive,
        len(available_negatives)
    )
    
    if sampling_strategy == "balanced":
        # Simple random sampling
        return np.random.choice(
            available_negatives["cve_id"].values,
            size=n_samples,
            replace=False
        ).tolist()
    
    elif sampling_strategy == "severity_weighted":
        # Sample based on base_score (higher scores more likely)
        if "base_score" in available_negatives.columns:
            scores = available_negatives["base_score"].fillna(0)  
            # Normalize to create sampling weights - handle case where all scores are 0
            if scores.sum() > 0:
                weights = scores / scores.sum()
            else:
                weights = None
        else:
            weights = None
            
        return np.random.choice(
            available_negatives["cve_id"].values,
            size=n_samples,
            replace=False,
            p=weights
        ).tolist()
    
    elif sampling_strategy == "temporal_matched":
        # Match publication dates of positives with negatives
        positive_cves_meta = features[features["cve_id"].isin(window_cves)]
        
        if len(positive_cves_meta) > 0:
            # Get publication date range of positive samples
            pub_date_min = positive_cves_meta["published_date"].min()
            pub_date_max = positive_cves_meta["published_date"].max()
            
            # Expand range by 6 months on each side
            pub_date_min = pub_date_min - pd.Timedelta(days=180)
            pub_date_max = pub_date_max + pd.Timedelta(days=180)
            
            # Filter negatives to similar publication timeframe
            temporal_negatives = available_negatives[
                (available_negatives["published_date"] >= pub_date_min) &
                (available_negatives["published_date"] <= pub_date_max)
            ]
            
            if len(temporal_negatives) >= n_samples:
                return np.random.choice(
                    temporal_negatives["cve_id"].values,
                    size=n_samples,
                    replace=False
                ).tolist()
        
        # Fallback to balanced sampling
        return negative_sampling(
            window_cves, all_cves, features, 
            window_start, window_end, n_negative_per_positive, "balanced"
        )
    
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


def generate_time_windows(
    df: pd.DataFrame,
    features: pd.DataFrame,
    window_size: int,
    prediction_horizon: int,
    stride: int,
    cve_mentions: Dict,
    sampling_strategy: str = "severity_weighted",
    min_positive_samples: int = 1
) -> pd.DataFrame:
    """
    Generate time windows using features from feature extraction.
    
    Args:
        df: Merged observations and features DataFrame
        features: CVE features DataFrame
        window_size: Size of observation window in days
        prediction_horizon: Prediction horizon in days
        stride: Stride between windows in days
        cve_mentions: Dictionary of CVE mentions
        sampling_strategy: Negative sampling strategy
        min_positive_samples: Minimum positive samples per window
    
    Returns:
        DataFrame with time windows and features
    """
    windows = []
    all_cves = set(features["cve_id"].unique())
    
    # Ensure we don't create windows that extend beyond our data
    max_date = df["date"].max()
    date_range = pd.date_range(
        df["date"].min(),
        max_date - pd.Timedelta(days=window_size + prediction_horizon),
        freq=f"{stride}D",
    )
    
    print(f"Generating {len(date_range)} time windows...")
    
    for i, start_date in enumerate(date_range):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing window {i+1}/{len(date_range)}")
            
        end_date = start_date + pd.Timedelta(days=window_size)
        target_start = end_date
        target_end = target_start + pd.Timedelta(days=prediction_horizon)

        # Get observations in current window
        window_obs = df[(df["date"] >= start_date) & (df["date"] < end_date)]
        
        if len(window_obs) < min_positive_samples:
            continue

        # Get CVEs with observations in this window (positive samples)
        window_cves = set(window_obs["cve_id"].unique())
        
        # Sample negative CVEs
        sampled_negative_cves = negative_sampling(
            window_cves=window_cves,
            all_cves=all_cves,
            features=features,
            window_start=start_date,
            window_end=end_date,
            n_negative_per_positive=6,
            sampling_strategy=sampling_strategy
        )

        # Process all CVEs (positive and negative)
        cves_to_process = list(window_cves) + sampled_negative_cves
        
        for cve in cves_to_process:
            # Get CVE metadata from features
            cve_meta = features[features["cve_id"] == cve]
            if len(cve_meta) == 0:
                continue
                
            cve_info = cve_meta.iloc[0].to_dict()
            
            # Get observations for this CVE in the window
            cve_window_obs = window_obs[window_obs["cve_id"] == cve]
            
            # Calculate target (was it exploited in prediction horizon?)
            future_obs = df[
                (df["date"] >= target_start) &
                (df["date"] < target_end) &
                (df["cve_id"] == cve)
            ]
            target = int(len(future_obs) > 0 and future_obs["count"].sum() > 0)
            
            # Basic observation features
            observations_in_window = len(cve_window_obs)
            total_count_in_window = cve_window_obs["count"].sum() if len(cve_window_obs) > 0 else 0
            
            # Time-based features using existing date columns
            days_since_published = (end_date - cve_info["published_date"]).days
            days_since_modified = (end_date - cve_info["last_modified_date"]).days
            
            # Mentions features (if available)
            if cve in cve_mentions:
                mentions_in_window = sum(
                    count for date, count in cve_mentions[cve].items() 
                    if start_date <= date < end_date
                )
                total_mentions = sum(cve_mentions[cve].values())
                mention_dates = [date for date in cve_mentions[cve].keys() if date < end_date]
                days_since_last_mention = (end_date - max(mention_dates)).days if mention_dates else None
                mention_frequency = mentions_in_window / window_size
            else:
                mentions_in_window = 0
                total_mentions = 0
                days_since_last_mention = None
                mention_frequency = 0
            
            # Create window-specific features
            window_features = {
                # Window metadata
                "window_start": start_date,
                "window_end": end_date,
                
                # Observation features
                "observations_in_window": observations_in_window,
                "total_count_in_window": total_count_in_window,
                "has_observations": int(observations_in_window > 0),
                
                # Time features from existing data
                "days_since_published": days_since_published,
                "days_since_modified": days_since_modified,
                
                # Mention features
                "mentions_in_window": mentions_in_window,
                "total_mentions": total_mentions,
                "days_since_last_mention": days_since_last_mention,
                "mention_frequency": mention_frequency,
                
                # Target
                "target": target
            }
            
            # Add all original CVE features from feature extraction
            cve_info.update(window_features)
            windows.append(cve_info)
                

    result_df = pd.DataFrame(windows)
    return result_df


def create_training_set(
    features_file: Path = None,
    observations_file: Path = None,
    cve_mentions_file: Path = None,
    output_file: Path = None,
    window_size: int = 30,
    prediction_horizon: int = 30,
    stride: int = 7,
    sampling_strategy: str = "severity_weighted",
    balance_ratio: float = 0.1
) -> Path:
    """
    Create training set from extracted features and observations.
    
    Args:
        features_file: Path to extracted features CSV (from feature_extraction.py)
        observations_file: Path to observations CSV
        cve_mentions_file: Path to CVE mentions JSON
        output_file: Path to save training set
        window_size: Size of observation window in days
        prediction_horizon: Prediction horizon in days
        stride: Stride between windows in days
        sampling_strategy: Negative sampling strategy
        balance_ratio: Ratio for balancing dataset
    
    Returns:
        Path to created training set file
    """
    # Use defaults if not provided
    if features_file is None:
        # Look for most recent features file
        features_files = list((DATA_DIR / "preprocessed").glob("nvd_features_*.csv"))
        if not features_files:
            raise FileNotFoundError("No features files found. Run feature extraction first.")
        features_file = max(features_files, key=lambda p: p.stat().st_mtime)
        print(f"Using features file: {features_file}")
    
    if observations_file is None:
        observations_file = OBSERVATIONS_PATH
    
    if cve_mentions_file is None:
        cve_mentions_file = CVE_MENTIONS_PATH
        
    if output_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATA_DIR / f"training_set_{timestamp}.csv"
    
    # Load data
    print("Loading data...")
    df, features = load_and_preprocess_data(observations_file, features_file)
    
    # Load CVE mentions if file exists
    cve_mentions = {}
    if cve_mentions_file.exists():
        cve_mentions = load_cve_mentions(cve_mentions_file)
    else:
        print("CVE mentions file not found, using empty mentions")
    
    print(f"Loaded {len(df)} observations and {len(features)} CVE records")
    print(f"Available CVE features: {list(features.columns)}")
    
    # Generate time windows
    print("Generating time windows...")
    feature_df = generate_time_windows(
        df=df,
        features=features,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        stride=stride,
        cve_mentions=cve_mentions,
        sampling_strategy=sampling_strategy,
        min_positive_samples=1
    )
    
    if len(feature_df) == 0:
        raise ValueError("No features generated")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(feature_df)}")
    print(f"Positive samples: {feature_df['target'].sum()}")
    print(f"Positive ratio: {feature_df['target'].mean():.2%}")
    
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file, index=False)
    print(f"Training set saved to {output_file}")
    
    return output_file
        


def create_inference_dataset(
    features_file: Path,
    output_dir: Path = None,
    window_size: int = 30
) -> Path:
    """
    Create inference dataset from extracted features (for prediction on new CVEs).
    
    Args:
        features_file: Path to extracted features CSV
        output_dir: Directory to save inference dataset
        window_size: Window size for feature generation
    
    Returns:
        Path to created inference dataset
    """
    if output_dir is None:
        output_dir = INFERENCE_DATASETS
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    features = pd.read_csv(features_file)
    features["published_date"] = pd.to_datetime(features["published_date"], utc=True)
    features["last_modified_date"] = pd.to_datetime(features["last_modified_date"], utc=True)
    
    # Use current date as reference
    reference_date = pd.Timestamp.now(tz='UTC')
    
    inference_data = []
    for _, cve_info in features.iterrows():
        # Calculate time-based features
        days_since_published = (reference_date - cve_info["published_date"]).days
        days_since_modified = (reference_date - cve_info["last_modified_date"]).days
        
        # Create inference features (no observation or mention data available)
        inference_features = {
            **cve_info.to_dict(),
            "days_since_published": days_since_published,
            "days_since_modified": days_since_modified,
            "observations_in_window": 0,  # New CVEs have no observations yet
            "total_count_in_window": 0,
            "has_observations": 0,
            "mentions_in_window": 0,
            "total_mentions": 0,
            "days_since_last_mention": None,
            "mention_frequency": 0,
            "reference_date": reference_date
        }
        
        inference_data.append(inference_features)
    
    # Create DataFrame and save
    inference_df = pd.DataFrame(inference_data)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"inference_dataset_{timestamp}.csv"
    
    inference_df.to_csv(output_file, index=False)
    print(f"Inference dataset saved to {output_file}")
    print(f"Created inference dataset with {len(inference_df)} CVEs")
    
    return output_file
        


if __name__ == "__main__":
    training_set_file = create_training_set()