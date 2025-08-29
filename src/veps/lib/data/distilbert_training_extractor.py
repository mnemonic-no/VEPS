import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd

from ..models.distilbert.base import load_json_data

class TrainingSetExtractor:
    """Extract and prepare training data from NVD JSON files."""
    
    def __init__(self, nvd_data_dir: Path, output_dir: Path):
        self.nvd_data_dir = nvd_data_dir
        self.output_dir = output_dir
        
        # Create output directories
        self.cvss_dir = output_dir / "cvss"
        self.cwe_dir = output_dir / "cwe"
        
        for directory in [self.cvss_dir, self.cwe_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def is_rejected_cve(self, cve_entry: Dict[str, Any]) -> bool:
        """Check if CVE is rejected and should be skipped."""
        try:
            vuln_status = cve_entry["cve"].get("vulnStatus", "")
            return vuln_status.lower() == "rejected"
        except (KeyError, AttributeError) as e:
            print(f"Error checking CVE status: {e}")
            return False

    def extract_description(self, cve_entry: Dict[str, Any]) -> Optional[str]:
        """Extract description from CVE entry."""
        try:
            descriptions = cve_entry["cve"]["descriptions"]
            if not descriptions:
                print(f"No descriptions found for CVE {cve_entry['cve']['id']}")
                return None
            
            # Look for English description first
            for desc in descriptions:
                if desc.get("lang") == "en":
                    return desc.get("value", "").strip()
            
            # Fallback to first description if no English found
            first_desc = descriptions[0].get("value", "").strip()
            if first_desc:
                return first_desc
            
            print(f"No valid description found for CVE {cve_entry['cve']['id']}")
            return None
            
        except (KeyError, IndexError, TypeError) as e:
            cve_id = cve_entry.get("cve", {}).get("id", "unknown")
            print(f"Error extracting description for CVE {cve_id}: {e}")
            return None

    def extract_cvss_data(self, cve_entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract CVSS metrics from CVE entry, prioritizing Primary source."""
        cve_id = cve_entry.get("cve", {}).get("id", "unknown")
        
        try:
            metrics = cve_entry.get("cve", {}).get("metrics", {})
            if not metrics:
                print(f"No metrics found for CVE {cve_id}")
                return None
            
            # Look for CVSS v3.1 metrics first, then v3.0
            cvss_metrics = metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30")
            
            if not cvss_metrics:
                print(f"No CVSS v3.x metrics found for CVE {cve_id}")
                return None
            
            # Find Primary metric first, fallback to first available
            primary_metric = None
            fallback_metric = None
            
            for metric in cvss_metrics:
                if metric.get("type") == "Primary":
                    primary_metric = metric
                    break
                elif fallback_metric is None:
                    fallback_metric = metric
            
            selected_metric = primary_metric or fallback_metric
            
            if not selected_metric:
                print(f"No valid CVSS metric found for CVE {cve_id}")
                return None
            
            cvss_data = selected_metric.get("cvssData", {})
            vector_string = cvss_data.get("vectorString", "")
            
            if not vector_string:
                print(f"No vector string found for CVE {cve_id}")
                return None
            
            # Parse CVSS vector: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H
            try:
                vector_parts = vector_string.split("/")
                if len(vector_parts) < 2:
                    raise ValueError("Invalid vector string format")
                
                metrics_dict = {}
                # Skip version part (CVSS:3.1)
                for metric_part in vector_parts[1:]:
                    if ":" not in metric_part:
                        continue
                    metric_name, metric_value = metric_part.split(":", 1)
                    metrics_dict[metric_name] = metric_value
                
                if not metrics_dict:
                    raise ValueError("No metrics parsed from vector string")
                
                # Add metadata
                metrics_dict["base_score"] = str(cvss_data.get("baseScore", ""))
                metrics_dict["severity"] = cvss_data.get("baseSeverity", "")
                metrics_dict["metric_type"] = selected_metric.get("type", "")
                
                return metrics_dict
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing CVSS vector for CVE {cve_id}: {e}")
                return None
            
        except (KeyError, TypeError, AttributeError) as e:
            print(f"Error extracting CVSS data for CVE {cve_id}: {e}")
            return None

    def extract_cwe_data(self, cve_entry: Dict[str, Any]) -> List[str]:
        """Extract CWE identifiers from CVE entry, prioritizing Primary source."""
        cve_id = cve_entry.get("cve", {}).get("id", "unknown")
        cwes = []
        
        try:
            weaknesses = cve_entry.get("cve", {}).get("weaknesses", [])
            
            # Process new weaknesses format
            primary_cwes = []
            fallback_cwes = []
            
            for weakness in weaknesses:
                weakness_type = weakness.get("type", "")
                descriptions = weakness.get("description", [])
                
                for desc in descriptions:
                    if desc.get("lang") == "en":
                        cwe_value = desc.get("value", "").strip()
                        if cwe_value.startswith("CWE-"):
                            if weakness_type == "Primary":
                                if cwe_value not in primary_cwes:
                                    primary_cwes.append(cwe_value)
                            else:
                                if cwe_value not in fallback_cwes:
                                    fallback_cwes.append(cwe_value)
            
            # Prefer primary CWEs, fallback to others
            cwes = primary_cwes if primary_cwes else fallback_cwes
            
        except (KeyError, TypeError, AttributeError) as e:
            print(f"Error extracting CWE data for CVE {cve_id}: {e}")
            
        return cwes

    def process_single_file(self, json_file: Path) -> Tuple[int, int]:
        """Process a single NVD JSON file and return counts of CVSS/CWE entries."""
        print(f"Processing {json_file}")
        
        try:
            cve_data = load_json_data(json_file)
            if not cve_data:
                print(f"Failed to load data from {json_file}")
                return 0, 0
            
            if "vulnerabilities" not in cve_data:
                print(f"No vulnerabilities found in {json_file}")
                return 0, 0
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return 0, 0
        
        cvss_count = 0
        cwe_count = 0
        skipped_count = 0
        
        for i, entry in enumerate(cve_data["vulnerabilities"]):
            try:
                # Check if CVE is rejected
                if self.is_rejected_cve(entry):
                    skipped_count += 1
                    continue
                
                cve_id = entry.get("cve", {}).get("id", f"unknown_entry_{i}")
                
                description = self.extract_description(entry)
                if not description:
                    print(f"Skipping CVE {cve_id}: No valid description")
                    continue
                
                # Process CVSS data
                cvss_metrics = self.extract_cvss_data(entry)
                if cvss_metrics:
                    try:
                        cvss_file_path = self.cvss_dir / f"{cve_id}.txt"
                        cvss_file_path.write_text(description, encoding="utf-8")
                        
                        cvss_entry = {
                            "filename": cvss_file_path.name,
                            "cve_id": cve_id,
                            **cvss_metrics,
                        }
                        self.cvss_entries.append(cvss_entry)
                        cvss_count += 1
                    except Exception as e:
                        print(f"Error saving CVSS data for {cve_id}: {e}")
                
                # Process CWE data
                cwes = self.extract_cwe_data(entry)
                if cwes:
                    try:
                        cwe_file_path = self.cwe_dir / f"{cve_id}.txt"
                        cwe_file_path.write_text(description, encoding="utf-8")
                        
                        cwe_entry = {
                            "filename": cwe_file_path.name,
                            "cve_id": cve_id,
                            "cwes": json.dumps(cwes),
                        }
                        self.cwe_entries.append(cwe_entry)
                        self.unique_cwes.update(cwes)
                        cwe_count += 1
                    except Exception as e:
                        print(f"Error saving CWE data for {cve_id}: {e}")
                        
            except Exception as e:
                cve_id = entry.get("cve", {}).get("id", f"entry_{i}")
                print(f"Error processing entry {cve_id}: {e}")
                continue
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} rejected CVEs")
                
        return cvss_count, cwe_count

    def extract_training_sets(self) -> Dict[str, int]:
        """Extract training sets from all NVD JSON files."""
        # Initialize storage
        self.cvss_entries: List[Dict[str, str]] = []
        self.cwe_entries: List[Dict[str, str]] = []
        self.unique_cwes: set = set()
        
        total_cvss = 0
        total_cwe = 0
        processed_files = 0
        
        # Process all JSON files
        json_files = list(self.nvd_data_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {self.nvd_data_dir}")
            return {}
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            try:
                cvss_count, cwe_count = self.process_single_file(json_file)
                total_cvss += cvss_count
                total_cwe += cwe_count
                processed_files += 1
            except Exception as e:
                print(f"Failed to process {json_file}: {e}")
                continue
        
        # Save metadata files
        try:
            self._save_metadata()
        except Exception as e:
            print(f"Error saving metadata: {e}")
        
        # Return statistics
        stats = {
            "processed_files": processed_files,
            "total_cvss_entries": total_cvss,
            "total_cwe_entries": total_cwe,
            "unique_cwes": len(self.unique_cwes),
        }
        
        self._log_statistics(stats)
        return stats

    def _save_metadata(self) -> None:
        """Save metadata CSV files."""
        # Save CVSS metadata
        if self.cvss_entries:
            try:
                cvss_df = pd.DataFrame(self.cvss_entries)
                cvss_metadata_path = self.cvss_dir / "cvss_metadata.csv"
                cvss_df.to_csv(cvss_metadata_path, index=False)
                print(f"Saved CVSS metadata: {cvss_metadata_path}")
            except Exception as e:
                print(f"Error saving CVSS metadata: {e}")
        else:
            print("No CVSS entries to save")
        
        # Save CWE metadata
        if self.cwe_entries:
            try:
                cwe_df = pd.DataFrame(self.cwe_entries)
                cwe_metadata_path = self.cwe_dir / "cwe_metadata.csv"
                cwe_df.to_csv(cwe_metadata_path, index=False)
                print(f"Saved CWE metadata: {cwe_metadata_path}")
            except Exception as e:
                print(f"Error saving CWE metadata: {e}")
        else:
            print("No CWE entries to save")

    def _log_statistics(self, stats: Dict[str, int]) -> None:
        """Log extraction statistics."""
        print("\n" + "="*50)
        print("Training Set Extraction Statistics:")
        print("="*50)
        print(f"  Processed files: {stats['processed_files']}")
        print(f"  CVSS entries: {stats['total_cvss_entries']}")
        print(f"  CWE entries: {stats['total_cwe_entries']}")
        print(f"  Unique CWEs: {stats['unique_cwes']}")
        print("="*50)

    def get_cwe_distribution(self) -> Dict[str, int]:
        """Get distribution of CWE labels for analysis."""
        if not hasattr(self, 'cwe_entries') or not self.cwe_entries:
            print("No CWE entries available for distribution analysis")
            return {}
            
        cwe_counts: Dict[str, int] = {}
        try:
            for entry in self.cwe_entries:
                cwes = json.loads(entry["cwes"])
                for cwe in cwes:
                    cwe_counts[cwe] = cwe_counts.get(cwe, 0) + 1
        except Exception as e:
            print(f"Error calculating CWE distribution: {e}")
            return {}
        
        return dict(sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True))

    def filter_rare_cwes(self, min_occurrences: int = 5) -> None:
        """Filter out CWEs that occur less than min_occurrences times."""
        if not hasattr(self, 'cwe_entries') or not self.cwe_entries:
            print("No CWE entries available for filtering")
            return
            
        try:
            cwe_distribution = self.get_cwe_distribution()
            if not cwe_distribution:
                print("No CWE distribution available for filtering")
                return
                
            rare_cwes = {cwe for cwe, count in cwe_distribution.items() if count < min_occurrences}
            
            if not rare_cwes:
                print("No rare CWEs found to filter")
                return
                
            print(f"Filtering {len(rare_cwes)} rare CWEs (< {min_occurrences} occurrences)")
            
            filtered_entries = []
            removed_files = 0
            
            for entry in self.cwe_entries:
                cwes = json.loads(entry["cwes"])
                filtered_cwes = [cwe for cwe in cwes if cwe not in rare_cwes]
                
                if filtered_cwes:  # Only keep entries with at least one common CWE
                    entry["cwes"] = json.dumps(filtered_cwes)
                    filtered_entries.append(entry)
                else:
                    # Remove the associated text file
                    file_path = self.cwe_dir / entry["filename"]
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            removed_files += 1
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
            
            self.cwe_entries = filtered_entries
            self.unique_cwes = self.unique_cwes - rare_cwes
            
            print(f"Kept {len(filtered_entries)} CWE entries after filtering")
            print(f"Removed {removed_files} text files for rare CWEs")
            
        except Exception as e:
            print(f"Error during CWE filtering: {e}")