# VEPS - Vulnerability Exploitation Prediction Score

A machine learning pipeline for predicting the likelihood of CVE exploitation using NVD data, CVE observations, and multiple models including DistilBERT and XGBoost.


## Overview

VEPS processes National Vulnerability Database (NVD) data to predict which CVEs are most likely to be exploited in the wild. The system combines:

- **Data Processing**: Automated NVD data download and feature extraction
- **DistilBERT Models**: For predicting missing CVSS vectors and CWE classifications
- **XGBoost Model**: For vulnerability exploitation likelihood prediction
- **Daily Pipeline**: Automated daily predictions on new CVE data

```
# Download latest NVD data
veps download --latest

# Extract features from NVD data
veps extract-features

# Create training dataset
veps build-trainset

# Perform hyperparameter search (optional)
veps tune

# Train vulnerability prediction model
veps train

# Make daily predictions
veps predict
```


## Required input files

`data/raw/` is immutable — do not write to it after initial ingestion.
See [`data/raw/README.md`](./data/raw/README.md) and
[`src/veps/data/schemas.py`](./src/veps/data/schemas.py) for the full
schema and committed sample files.

CVE Observations (`data/raw/cve_observations.csv`)
```
cve,date,count
CVE-2021-44228,2021-12-10,15
CVE-2021-44228,2021-12-11,42
```

CVE mentions (`data/raw/cve_mentions.json`) (optional)
```
{
  "CVE-2021-44228": {
    "2021-12-09": 5,
    "2021-12-10": 23
  }
}
```

## Output Format
Daily predictions saved as `predictions_YYYYMMDD.csv`
```
cve_id,base_score,attack_vector,...,exploitation_probability,percentile
CVE-2024-12345,9.8,NETWORK,...,0.94,99.2
```



# Cyberrisk
VEPS was developed in the Cyberrisk project.

The research project “Cyberrisk” is a joint partnership between mnemonic, [Norsk Regnesentral](https://nr.no/), [Avinor](https://avinor.no/) and [DNB](https://www.dnb.no/).

[The Research Council of Norway](https://www.forskningsradet.no/en/) has granted the research project as part of their [Innovation Project for the Industrial Sector](https://www.forskningsradet.no/en/call-for-proposals/2021/innovation-project-for-the-industrial-sector/), a funding instrument that provides grants to business-led innovation projects that make extensive use of research and development (R&D). The Innovation Project is to lead to renewal and sustainable value creation for the project’s business partners, and funding should also generate socioeconomic benefits by making new knowledge and solutions available.