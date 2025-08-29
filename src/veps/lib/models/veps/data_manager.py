import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from .config import TRAINED_MODEL_DIR  # Updated import
from ....config import DATA_DIR  # Main config

def load_dataset(dataset_path=None):
    """Load dataset with flexible path handling."""
    if dataset_path is None:
        # Look for most recent training set or features file
        training_files = list((DATA_DIR / "preprocessed").glob("training_set_*.csv"))
        if training_files:
            dataset_path = max(training_files, key=lambda p: p.stat().st_mtime)
        else:
            dataset_path = DATA_DIR / "features.csv"
    
    df = pd.read_csv(dataset_path)
    date_columns = [
        "window_start",
        "window_end", 
        "published_date",
        "last_modified_date",
    ]
    for col in date_columns:
        if col in df.columns:  # Only process existing columns
            df[col] = pd.to_datetime(df[col], format='ISO8601')
    return df

def split_data_by_date(df: pd.DataFrame, cutoff_date):
    """Split the data into train and test sets based on a cutoff date."""
    train = df[df["window_end"] < cutoff_date]
    test = df[df["window_end"] >= cutoff_date]
    return train, test

def save_pipeline(pipeline: Pipeline):
    """Save trained pipeline."""
    save_file_name = "vuln_prediction.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    joblib.dump(pipeline, save_path)
    print(f"Pipeline saved to {save_path}")

def load_pipeline(file_name: str = "vuln_prediction.pkl"):
    """Load trained pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Model not found: {file_path}")
    return joblib.load(filename=file_path)

def fill_with_predictions(df, use_predicted_cwe=False, use_predicted_cvss=False):
    """Fill missing values with predictions from distilbert model."""
    df_updated = df.copy()
    
    if use_predicted_cwe and 'predicted_cwe_list' in df.columns:
        df_updated.loc[df_updated['cwe_list'].isin(['[]', '', None]), 'cwe_list'] = \
            df_updated.loc[df_updated['cwe_list'].isin(['[]', '', None]), 'predicted_cwe_list']
    
    if use_predicted_cvss:
        cvss_fields = [
            'vector_string', 'attack_vector', 'attack_complexity', 'privileges_required',
            'user_interaction', 'scope', 'confidentiality_impact', 'integrity_impact',
            'availability_impact', 'base_score', 'base_severity', 'exploitability_score',
            'impact_score'
        ]
        
        for field in cvss_fields:
            if field in df.columns and f'predicted_{field}' in df.columns:
                mask = df_updated[field].isna() | (df_updated[field] == '')
                df_updated.loc[mask, field] = df_updated.loc[mask, f'predicted_{field}']
    
    return df_updated