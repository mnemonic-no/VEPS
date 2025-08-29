# VEPS - Vulnerability Exploitation Prediction Score

A machine learning pipeline for predicting the likelihood of CVE exploitation using NVD data, CVE observations, and multiple models including DistilBERT and XGBoost.


## Overview

VEPS processes National Vulnerability Database (NVD) data to predict which CVEs are most likely to be exploited in the wild. The system combines:

- **Data Processing**: Automated NVD data download and feature extraction
- **DistilBERT Models**: For predicting missing CVSS scores and CWE classifications
- **XGBoost Model**: For vulnerability exploitation likelihood prediction
- **Daily Pipeline**: Automated daily predictions on new CVE data

```
# Download latest NVD data
uv run src/scripts/data/download_data.py --latest

# Extract features from NVD data
uv run src/scripts/veps/extract_features.py

# Create training dataset
uv run src/scripts/veps/create_training_set.py

# Perform hyperparameter search (optional)
uv run src/scripts/veps/hyperparameter_tuning.py

# Train vulnerability prediction model
uv run src/scripts/veps/train_pipeline.py

# Make daily predictions
uv run src/scripts/veps/prediction.py
```


## Required input files
CVE Observations(`data/cve_observations.csv`)
```
cve,date,count
CVE-2021-44228,2021-12-10,15
CVE-2021-44228,2021-12-11,42
```

CVE mentions (`data/cve_mentions.json`) (optional)
```
{
  "CVE-2021-44228": {
    "2021-12-09": 5,
    "2021-12-10": 23
  }
}
```

Output Format
Daily predictions saved as `predictions_YYYYMMDD.csv`
```
cve_id,base_score,attack_vector,...,exploitation_probability,percentile
CVE-2024-12345,9.8,NETWORK,...,0.94,99.2
```