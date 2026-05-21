from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from . import paths


class FeatureColumns(BaseModel):
    cwe_features: Optional[List[str]] = []
    log_features: Optional[List[str]] = []
    other_numeric_features: Optional[List[str]] = []
    categorical_features: Optional[List[str]] = []
    passthrough: Optional[List[str]] = []


class Preprocessing(BaseModel):
    window_size: int = 30
    prediction_horizon: int = 30
    stride_days: int = 30
    min_positive_samples: int = 1
    # None resolves to prediction_horizon at use time. Override to widen
    # the gap (e.g. for slow-to-observe exploits).
    holdout_gap_days: Optional[int] = None

    def resolved_holdout_gap_days(self) -> int:
        return (
            self.holdout_gap_days
            if self.holdout_gap_days is not None
            else self.prediction_horizon
        )


class XGBoostConfig(BaseModel):
    """Structural XGBoost settings only. Tuned hyperparameters (learning_rate,
    n_estimators, max_depth, etc.) are NOT stored here — they are persisted by
    `veps tune` to a best_params.json artifact and loaded by `veps train`."""

    tree_method: Literal["auto", "hist", "approx"] = "hist"
    enable_categorical: bool = True
    max_cat_to_onehot: int = 4
    device: Literal["cpu", "cuda"] = "cpu"
    random_state: int = 42


class CalibrationConfig(BaseModel):
    """Post-fit probability calibration via a held-out slice carved between
    train and the holdout gap. See ``split_data_for_calibration``."""

    enabled: bool = True
    method: Literal["isotonic", "sigmoid"] = "isotonic"
    slice_days: int = 60


class TuneConfig(BaseModel):
    """Optuna study persistence. Override per-run via `veps tune --study-name`
    / `--storage`."""

    study_name: str = "veps_phase3"
    storage: str = "sqlite:///veps_study.db"


class DistilBertHyperparameters(BaseModel):
    model_name: str = "distilbert-base-uncased"
    num_epochs: int = 5
    batch_size_cvss: int = 32
    batch_size_cwe: int = 64
    learning_rate: float = 2e-5
    max_length: int = 512
    patience: int = 2
    seeds: List[int] = Field(default_factory=lambda: [42])
    test_months: int = 6
    val_months: int = 3
    num_workers: int = 2
    amp_enabled: bool = True
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip_max_norm: float = 1.0
    head_loss_weighting: Literal["equal", "log_k"] = "log_k"
    class_weight_enabled: bool = True
    class_weight_clip: float = 5.0
    # BCE pos_weight = neg/pos lives in a wider range than CE class weights
    # (rare CWEs can easily land at 50+), so it gets its own clip.
    cwe_pos_weight_clip: float = 10.0
    cwe_keep_top_n: int = 15
    cvss_selection_metric: Literal["mean_macro_f1", "vector_exact_match"] = "mean_macro_f1"


class UsePredictions(BaseModel):
    cwe: Optional[bool] = False
    cvss: Optional[bool] = False


class Config(BaseModel):
    feature_columns: Optional[FeatureColumns] = FeatureColumns()
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    distilbert: DistilBertHyperparameters = Field(
        default_factory=DistilBertHyperparameters
    )
    tune: TuneConfig = Field(default_factory=TuneConfig)
    use_predictions: Optional[UsePredictions] = UsePredictions()
    target: str = "target"
    # Absolute holdout cutoff (ISO date). Mutually exclusive with holdout_days.
    cutoff_date: Optional[str] = None
    # Relative holdout cutoff: last N days of data become holdout. Anchored on
    # max(window_end) so the split moves with the data tail.
    holdout_days: Optional[int] = None

    @model_validator(mode="after")
    def _check_cutoff_exclusive(self) -> "Config":
        if self.cutoff_date is not None and self.holdout_days is not None:
            raise ValueError(
                "Set either cutoff_date (absolute) or holdout_days (relative), "
                "not both."
            )
        if self.holdout_days is not None and self.holdout_days <= 0:
            raise ValueError(
                f"holdout_days must be > 0, got {self.holdout_days}"
            )
        return self


def find_config_file() -> Path:
    if paths.CONFIG_FILE_PATH.is_file():
        return paths.CONFIG_FILE_PATH
    raise Exception(f"Config not found in {paths.CONFIG_FILE_PATH}")


def fetch_config_from_yaml() -> Config:
    cfg_path = find_config_file()

    with open(cfg_path, "r") as conf_file:
        yaml_data = yaml.safe_load(conf_file)
        return Config(**yaml_data)
