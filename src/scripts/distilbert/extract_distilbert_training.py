import sys
import argparse
from pathlib import Path


from veps.lib.data.distilbert_training_extractor import TrainingSetExtractor
from veps.config import NVD_FILEPATH, TRAINING_DIR

def main():
    parser = argparse.ArgumentParser(description="Extract training data from NVD files")
    parser.add_argument("--input-dir", type=Path, help="Directory containing NVD JSON files")
    parser.add_argument("--output-dir", type=Path, help="Output directory for training data")
    parser.add_argument("--filter-rare-cwes", type=int, default=0, 
                       help="Filter CWEs with fewer than N occurrences (0 = no filtering)")
    parser.add_argument("--show-cwe-distribution", action="store_true",
                       help="Show CWE distribution statistics")
    
    args = parser.parse_args()

    if not NVD_FILEPATH.exists():
        print(f"Input directory does not exist: {NVD_FILEPATH}")
        return 1
    
    extractor = TrainingSetExtractor(NVD_FILEPATH, TRAINING_DIR)
    stats = extractor.extract_training_sets()
    
    if args.filter_rare_cwes > 0:
        extractor.filter_rare_cwes(args.filter_rare_cwes)
        extractor._save_metadata()
    
    if args.show_cwe_distribution:
        print("\nCWE Distribution (top 20):")
        distribution = extractor.get_cwe_distribution()
        for i, (cwe, count) in enumerate(list(distribution.items())[:20]):
            print(f"  {cwe}: {count}")
        if len(distribution) > 20:
            print(f"  ... and {len(distribution) - 20} more")
    
    print(f"\nTraining data extraction completed!")
    print(f"Output directory: {TRAINING_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())