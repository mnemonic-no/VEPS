import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime

from ...config import NVD_FILEPATH, PREPROCESS_FILEPATH


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Load and return JSON data from a file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
    except IOError:
        print(f"Error reading file: {file_path}")
    return None


def extract_data_from_cve(cve_entry: Dict) -> Dict[str, Any]:
    """Extract relevant data from a CVE entry."""
    cve_data = cve_entry.get("cve", {})

    return {
        "cve_id": cve_data.get("id"),
        "published_date": cve_data.get("published"),
        "last_modified_date": cve_data.get("lastModified"),
        "cwe_list": extract_cwe_list(cve_entry),
        "predicted_cwe_list": extract_cwe_list(cve_entry, predicted=True),
        **count_tags_in_references(cve_entry),
        **extract_cvss_details(cve_entry, "original"),
        **extract_cvss_details(cve_entry, "predicted"),
    }


def extract_cwe_list(cve_entry: Dict, predicted: bool = False) -> List[str]:
    """Extract CWE list from a CVE entry."""
    weaknesses = cve_entry.get("cve", {}).get("weaknesses", [])
    cwe_list = []
    
    if predicted:
        # Look for predicted CWE data
        for weakness in weaknesses:
            if weakness.get("type") == "Predicted":
                descriptions = weakness.get("description", [])
                for desc in descriptions:
                    value = desc.get("value", "")
                    if value and value != "NVD-CWE-Other":
                        cwe_list.append(value)
                break  # Only take first predicted entry
    else:
        # Extract primary CWE data
        for weakness in weaknesses:
            if weakness.get("type") == "Primary":
                descriptions = weakness.get("description", [])
                for desc in descriptions:
                    value = desc.get("value", "")
                    if value and value != "NVD-CWE-Other":
                        cwe_list.append(value)
                break  # Only take primary
        
        # If no primary found, take the first non-predicted one
        if not cwe_list:
            for weakness in weaknesses:
                if weakness.get("type") != "Predicted":
                    descriptions = weakness.get("description", [])
                    for desc in descriptions:
                        value = desc.get("value", "")
                        if value and value != "NVD-CWE-Other":
                            cwe_list.append(value)
                    break
    
    return cwe_list


def extract_cvss_details(cve_entry: Dict, cvss_type: str) -> Dict[str, Any]:
    """Extract CVSS details from a CVE entry."""
    metrics = cve_entry.get("cve", {}).get("metrics", {})
    cvss_v31_metrics = metrics.get("cvssMetricV31", [])
    
    if not cvss_v31_metrics:
        return {}
    
    target_metric = None
    
    if cvss_type == "predicted":
        # Look for predicted CVSS data
        for metric in cvss_v31_metrics:
            if metric.get("type") == "Predicted":
                target_metric = metric
                break
    else:
        # Find primary metric, or use first non-predicted one
        for metric in cvss_v31_metrics:
            if metric.get("type") == "Primary":
                target_metric = metric
                break
        
        if not target_metric:
            # Use first non-predicted metric
            for metric in cvss_v31_metrics:
                if metric.get("type") != "Predicted":
                    target_metric = metric
                    break
    
    if not target_metric:
        return {}
    
    cvss_data = target_metric.get("cvssData", {})
    prefix = "predicted_" if cvss_type == "predicted" else ""
    
    return {
        f"{prefix}vector_string": cvss_data.get("vectorString"),
        f"{prefix}attack_vector": cvss_data.get("attackVector"),
        f"{prefix}attack_complexity": cvss_data.get("attackComplexity"),
        f"{prefix}privileges_required": cvss_data.get("privilegesRequired"),
        f"{prefix}user_interaction": cvss_data.get("userInteraction"),
        f"{prefix}scope": cvss_data.get("scope"),
        f"{prefix}confidentiality_impact": cvss_data.get("confidentialityImpact"),
        f"{prefix}integrity_impact": cvss_data.get("integrityImpact"),
        f"{prefix}availability_impact": cvss_data.get("availabilityImpact"),
        f"{prefix}base_score": cvss_data.get("baseScore"),
        f"{prefix}base_severity": cvss_data.get("baseSeverity"),
        f"{prefix}exploitability_score": target_metric.get("exploitabilityScore"),
        f"{prefix}impact_score": target_metric.get("impactScore"),
    }


def count_tags_in_references(cve_entry: Dict) -> Dict[str, int]:
    """Count occurrences of each tag in references."""
    references = cve_entry.get("cve", {}).get("references", [])
    tag_counts: Dict[str, int] = {}
    print(references)

    for reference in references:
        for tag in reference.get("tags", []):
            ref_tag = "ref_" + tag.lower().replace(" ", "_").replace("/", "_")
            tag_counts[ref_tag] = tag_counts.get(ref_tag, 0) + 1

    return tag_counts


def count_tags_and_domains_in_references(cve_entry: Dict) -> Dict[str, int]:
    """Count occurrences of each tag and domain in references, using tags as domain prefixes."""
    references = cve_entry.get("cve", {}).get("references", [])
    counts: Dict[str, int] = {}

    for reference in references:
        tags = reference.get("tags", [])
        for tag in tags:
            ref_tag = "ref_" + tag.lower().replace(" ", "_").replace("/", "_")
            counts[ref_tag] = counts.get(ref_tag, 0) + 1

        # Count domains with tag prefixes
        url = reference.get("url", "")
        if url:
            try:
                domain = urlparse(url).netloc.lower()
                if domain:
                    domain_key = "domain_" + domain.replace(".", "_")
                    counts[domain_key] = counts.get(domain_key, 0) + 1

                    for tag in tags:
                        tag_prefix = tag.lower().replace(" ", "_").replace("/", "_")
                        tagged_domain_key = (
                            f"domain_{tag_prefix}_{domain.replace('.', '_')}"
                        )
                        counts[tagged_domain_key] = counts.get(tagged_domain_key, 0) + 1
            except ValueError:
                continue

    return counts


def is_rejected_cve(cve_entry: Dict) -> bool:
    """Check if a CVE entry is rejected."""
    vuln_status = cve_entry.get("cve", {}).get("vulnStatus", "")
    return vuln_status == "Rejected"


def process_cve_entries(nvd_filepath: Path) -> List[Dict]:
    """Process all CVE entries from JSON files in the given directory."""
    cve_data = []
    for file_path in sorted(nvd_filepath.iterdir()):
        if file_path.suffix == '.json':  # Only process JSON files
            print(f"Processing: {file_path}")
            corpus = load_json_file(file_path)
            if corpus:
                cve_data.extend(
                    extract_data_from_cve(cve_entry)
                    for cve_entry in corpus.get("vulnerabilities", [])
                    if not is_rejected_cve(cve_entry)
                )
    return cve_data


def extract_features_from_nvd(nvd_files: List[Path] = None, output_file: Path = None) -> Path:
    """
    Extract features from NVD JSON files and save to CSV.
    
    Args:
        nvd_files: List of specific NVD JSON files to process. If None, processes all files in NVD_FILEPATH.
        output_file: Path where to save the CSV file. If None, generates timestamped file in PREPROCESS_FILEPATH.
    
    Returns:
        Path to the generated CSV file.
    """
    # Ensure output directory exists
    PREPROCESS_FILEPATH.mkdir(parents=True, exist_ok=True)
    
    # Process specific files or all files in NVD directory
    if nvd_files:
        cve_data = []
        for file_path in nvd_files:
            if file_path.suffix == '.json':
                print(f"Processing: {file_path}")
                corpus = load_json_file(file_path)
                if corpus:
                    cve_data.extend(
                        extract_data_from_cve(cve_entry)
                        for cve_entry in corpus.get("vulnerabilities", [])
                        if not is_rejected_cve(cve_entry)
                    )
    else:
        # Process all files in the NVD directory
        cve_data = process_cve_entries(NVD_FILEPATH)
    
    # Create DataFrame
    df = pd.DataFrame(cve_data)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PREPROCESS_FILEPATH / f"nvd_features_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data has been processed and saved to '{output_file}'")
    print(f"Processed {len(cve_data)} CVE entries")
    
    return output_file


def extract_features_from_directory(input_dir: Path, output_file: Path) -> Path:
    """
    Extract features from all NVD JSON files in a directory.
    
    Args:
        input_dir: Directory containing NVD JSON files
        output_file: Path where to save the CSV file
    
    Returns:
        Path to the generated CSV file.
    """
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cve_data = process_cve_entries(input_dir)
    df = pd.DataFrame(cve_data)
    df.to_csv(output_file, index=False)
    print(f"Data has been processed and saved to '{output_file}'")
    print(f"Processed {len(cve_data)} CVE entries")
    
    return output_file


def extract_latest_features() -> Path:
    """Extract features from the most recent NVD files."""
    nvd_files = list(NVD_FILEPATH.glob("*.json"))
    if not nvd_files:
        raise FileNotFoundError(f"No JSON files found in {NVD_FILEPATH}")
    
    # Sort by modification time and take the most recent ones (last 2 files)
    recent_files = sorted(nvd_files, key=lambda p: p.stat().st_mtime, reverse=True)[:2]
    print(f"Processing {len(recent_files)} most recent files")
    
    return extract_features_from_nvd(recent_files)
