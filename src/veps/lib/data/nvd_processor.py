import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..models.distilbert.predict import CombinedDistilBertClassifier


def load_json_data(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


class NVDProcessor:
    """Process NVD JSON files and add ML predictions."""
    
    def __init__(self, models_dir: Path, model_name: str = "distilbert-base-uncased"):
        self.predictor = CombinedDistilBertClassifier(models_dir, model_name)

    def extract_description(self, cve_item: Dict[str, Any]) -> str:
        """Extract description from CVE item."""
        descriptions = cve_item.get("cve", {}).get("descriptions", [])
        
        # Look for English description first
        for desc in descriptions:
            if desc.get("lang") == "en":
                return desc.get("value", "")
        
        # If no English description, take the first one
        if descriptions:
            return descriptions[0].get("value", "")
        
        return ""

    def add_predictions_to_cve(self, cve_item: Dict[str, Any]) -> Dict[str, Any]:
        """Add ML predictions to a single CVE item."""
        description = self.extract_description(cve_item)
        
        if not description:
            print(f"No description found for CVE: {cve_item.get('cve', {}).get('id', 'Unknown')}")
            return cve_item

        try:
            predictions = self.predictor.predict_for_cve(description)
            
            if "cve" not in cve_item:
                cve_item["cve"] = {}
            if "weaknesses" not in cve_item["cve"]:
                cve_item["cve"]["weaknesses"] = []

            predicted_weakness = {
                "source": "ml_prediction",
                "type": "Predicted",
                "description": [
                    {"lang": "en", "value": cwe} 
                    for cwe in predictions["cwes"]
                ]
            }
            cve_item["cve"]["weaknesses"].append(predicted_weakness)

            if "metrics" not in cve_item["cve"]:
                cve_item["cve"]["metrics"] = {}
            if "cvssMetricV31" not in cve_item["cve"]["metrics"]:
                cve_item["cve"]["metrics"]["cvssMetricV31"] = []
            
            cvss_prediction = predictions["cvss"]
            if "error" not in cvss_prediction:
                predicted_cvss = {
                    "source": "ml_prediction",
                    "type": "Predicted",
                    "cvssData": cvss_prediction["cvssData"],
                    "exploitabilityScore": cvss_prediction["exploitabilityScore"],
                    "impactScore": cvss_prediction["impactScore"]
                }
                cve_item["cve"]["metrics"]["cvssMetricV31"].append(predicted_cvss)
            
        except Exception as e:
            print(f"Error predicting for CVE {cve_item.get('cve', {}).get('id', 'Unknown')}: {e}")

        return cve_item

    def process_nvd_file(self, input_file: Path, output_file: Path) -> None:
        """Process a single NVD JSON file."""
        print(f"Processing: {input_file}")
        
        nvd_data = load_json_data(input_file)
        if not nvd_data:
            print(f"Failed to load data from {input_file}")
            return

        processed_count = 0
        for cve_item in nvd_data.get("vulnerabilities", []):
            self.add_predictions_to_cve(cve_item)
            processed_count += 1

        # Save processed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(nvd_data, f, indent=2)
        
        print(f"Processed {processed_count} CVEs from {input_file} -> {output_file}")

    def process_directory(self, input_dir: Path, output_dir: Path) -> List[Path]:
        """Process all JSON files in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_files = []
        
        for json_file in input_dir.glob("*.json"):
            try:
                output_file = output_dir / json_file.name
                self.process_nvd_file(json_file, output_file)
                processed_files.append(output_file)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"Processed {len(processed_files)} files from {input_dir}")
        return processed_files