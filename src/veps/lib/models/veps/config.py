from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List
import yaml

from ....config import DATA_DIR, MODELS_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
TRAINED_MODEL_DIR = MODELS_DIR / "vuln_pred"
CONFIG_FILE_PATH = PROJECT_ROOT.parent / "config" / "config.yaml"

class FeatureColumns(BaseModel):
    cwe_features: Optional[List[str]] = []
    log_features: Optional[List[str]] = []
    other_numeric_features: Optional[List[str]] = []
    categorical_features: Optional[List[str]] = []
    passthrough: Optional[List[str]] = []

class Hyperparameters(BaseModel):
    learning_rate: float = 0.3
    n_estimators: int = 100         
    max_depth: int = 6         
    min_child_weight: int = 1    
    subsample: float = 1.0       
    colsample_bytree: float = 1.0  
    gamma: float = 0.0             
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0 

class UsePredictions(BaseModel):
    cwe: Optional[bool] = False
    cvss: Optional[bool] = False

class Config(BaseModel):
    feature_columns: Optional[FeatureColumns] = FeatureColumns()
    hyperparameters: Optional[Hyperparameters] = Hyperparameters()
    use_predictions: Optional[UsePredictions] = UsePredictions()
    training_data_file: str
    target: str = "target"
    pipeline_name: str = "vuln_prediction"
    pipeline_save_file: str = "vuln_prediction.pkl"
    cutoff_date: Optional[str] = None

def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found in {CONFIG_FILE_PATH}")

def fetch_config_from_yaml() -> dict:
    cfg_path = find_config_file()
    
    with open(cfg_path, "r") as conf_file:
        yaml_data = yaml.safe_load(conf_file)
        return Config(**yaml_data).model_dump()