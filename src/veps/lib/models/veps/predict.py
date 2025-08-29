from typing import Union
import pandas as pd
import joblib
from pathlib import Path
from .data_manager import load_pipeline, fill_with_predictions
from .config import fetch_config_from_yaml

def predict_vulnerabilities(
    input_data: Union[str, Path, pd.DataFrame], 
    model_dir: Path = None,
    output_file: Path = None,
    use_predicted_cwe: bool = False,
    use_predicted_cvss: bool = False
) -> pd.DataFrame:
    """
    Make vulnerability predictions.
    
    Args:
        input_data: Input data as file path or DataFrame
        model_dir: Directory containing the trained model (optional)
        output_file: Path to save results (optional)
        use_predicted_cwe: Whether to use predicted CWE values (optional)
        use_predicted_cvss: Whether to use predicted CVSS values (optional)
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    if isinstance(input_data, (str, Path)):
        data = pd.read_csv(input_data)
    else:
        data = input_data.copy()
    

    config = fetch_config_from_yaml()
    if use_predicted_cwe is None:
        use_predicted_cwe = config.get('use_predictions', {}).get('cwe', False)
    if use_predicted_cvss is None:
        use_predicted_cvss = config.get('use_predictions', {}).get('cvss', False)

    
    # Fill with predictions if requested
    if use_predicted_cwe or use_predicted_cvss:
        data = fill_with_predictions(data, use_predicted_cwe, use_predicted_cvss)
    
    # Load model
    if model_dir:
        model_path = model_dir / "vuln_prediction.pkl"
        classifier = joblib.load(model_path)
    else:
        classifier = load_pipeline("vuln_prediction.pkl")
    
    # Make predictions
    probabilities = classifier.predict_proba(data)[:, 1]
    data['exploitation_probability'] = probabilities
    data['percentile'] = data['exploitation_probability'].rank(pct=True)
    
    # Save if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_file, index=False)
    
    return data