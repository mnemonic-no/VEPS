import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..distilbert.predict import CombinedDistilBertClassifier
from ..paths import CORPUS_FILEPATH, MODELS_DIR, NVD_FILEPATH, PREDICTION_CACHE
from .prediction_cache import PredictionCache


def load_json_data(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


class NVDEnricher:
    """Enrich NVD JSON entries with DistilBERT-predicted CVSS/CWE labels.

    A :class:`PredictionCache` can be passed in to skip BERT for CVEs whose
    ``cve.lastModified`` matches a previously stored entry. The classifier is
    loaded lazily so a full-cache run never pays the model load cost.
    """

    def __init__(
        self,
        models_dir: Path,
        model_name: str = "distilbert-base-uncased",
        cache: Optional[PredictionCache] = None,
    ):
        self.models_dir = models_dir
        self.model_name = model_name
        self.cache = cache
        self._predictor: Optional[CombinedDistilBertClassifier] = None

    @property
    def predictor(self) -> CombinedDistilBertClassifier:
        if self._predictor is None:
            self._predictor = CombinedDistilBertClassifier(
                self.models_dir, self.model_name
            )
        return self._predictor

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

    def _compute_predictions(self, description: str) -> Dict[str, Any]:
        """Run BERT and return ``{"cwes": [...], "cvss": {...}}``."""
        result = self.predictor.predict_for_cve(description)
        return {"cwes": result["cwes"], "cvss": result["cvss"]}

    def _apply_predictions(
        self,
        cve_item: Dict[str, Any],
        predictions: Dict[str, Any],
    ) -> None:
        """Inject the cached or freshly computed predictions into ``cve_item``."""
        if "cve" not in cve_item:
            cve_item["cve"] = {}
        if "weaknesses" not in cve_item["cve"]:
            cve_item["cve"]["weaknesses"] = []

        predicted_weakness = {
            "source": "ml_prediction",
            "type": "Predicted",
            "description": [
                {"lang": "en", "value": cwe} for cwe in predictions["cwes"]
            ],
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
                "impactScore": cvss_prediction["impactScore"],
            }
            cve_item["cve"]["metrics"]["cvssMetricV31"].append(predicted_cvss)

    def add_predictions_to_cve(self, cve_item: Dict[str, Any]) -> str:
        """Add ML predictions to a CVE item.

        Returns ``"hit"`` if the prediction came from the cache, ``"miss"`` if
        BERT was invoked, or ``"skip"`` when the CVE had no description / an
        error blocked enrichment.
        """
        description = self.extract_description(cve_item)
        cve_id = cve_item.get("cve", {}).get("id", "Unknown")

        if not description:
            print(f"No description found for CVE: {cve_id}")
            return "skip"

        last_modified = cve_item.get("cve", {}).get("lastModified")

        if self.cache is not None and last_modified is not None:
            hit = self.cache.get(cve_id)
            if hit is not None and hit[0] == last_modified:
                self._apply_predictions(cve_item, hit[1])
                return "hit"

        try:
            predictions = self._compute_predictions(description)
        except Exception as e:
            print(f"Error predicting for CVE {cve_id}: {e}")
            return "skip"

        self._apply_predictions(cve_item, predictions)

        if self.cache is not None and last_modified is not None:
            self.cache.put(cve_id, last_modified, predictions)

        return "miss"

    def process_nvd_file(self, input_file: Path, output_file: Path) -> None:
        """Process a single NVD JSON file."""
        print(f"Processing: {input_file}")

        nvd_data = load_json_data(input_file)
        if not nvd_data:
            print(f"Failed to load data from {input_file}")
            return

        counts = {"hit": 0, "miss": 0, "skip": 0}
        for cve_item in nvd_data.get("vulnerabilities", []):
            counts[self.add_predictions_to_cve(cve_item)] += 1

        # Save processed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(nvd_data, f, indent=2)

        total = counts["hit"] + counts["miss"] + counts["skip"]
        if self.cache is not None:
            self.cache.flush()
            print(
                f"Processed {total} CVEs from {input_file} -> {output_file} "
                f"(cache hits={counts['hit']}, misses={counts['miss']}, "
                f"skipped={counts['skip']})"
            )
        else:
            print(f"Processed {total} CVEs from {input_file} -> {output_file}")

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


def main(args):
    models_dir = MODELS_DIR / 'distilbert'

    cache_path = getattr(args, "cache_path", None) or PREDICTION_CACHE
    use_cache = not getattr(args, "no_cache", False)

    cache_cm: Optional[PredictionCache] = None
    if use_cache:
        cache_cm = PredictionCache(cache_path)
        cache_cm.open()
        if getattr(args, "rebuild_cache", False):
            print(f"Rebuilding prediction cache at {cache_path}")
            cache_cm.clear()
        else:
            print(f"Using prediction cache at {cache_path} (entries={cache_cm.size()})")

    try:
        enricher = NVDEnricher(models_dir, cache=cache_cm)

        if args.single_file:
            input_file = args.single_file
            output_file = (args.output_dir or CORPUS_FILEPATH) / input_file.name
            enricher.process_nvd_file(input_file, output_file)
        else:
            input_dir = args.input_dir or NVD_FILEPATH
            output_dir = args.output_dir or CORPUS_FILEPATH

            processed_files = enricher.process_directory(input_dir, output_dir)
            print(f"Successfully processed {len(processed_files)} files")
    finally:
        if cache_cm is not None:
            cache_cm.close()
