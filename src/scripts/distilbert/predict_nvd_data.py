
import argparse
from pathlib import Path

from veps.lib.data.nvd_processor import NVDProcessor
from veps.config import MODELS_DIR, DATA_DIR, NVD_FILEPATH, CORPUS_FILEPATH

def main():
    parser = argparse.ArgumentParser(description="Add ML predictions to NVD data")
    parser.add_argument("--input-dir", type=Path, help="Input directory with NVD JSON files")
    parser.add_argument("--output-dir", type=Path, help="Output directory for processed files")
    parser.add_argument("--single-file", type=Path, help="Process a single file")
    
    args = parser.parse_args()
    models_dir = MODELS_DIR / 'distilbert'
    processor = NVDProcessor(models_dir)
    
    if args.single_file:
        input_file = args.single_file
        output_file = (args.output_dir or DATA_DIR / "processed") / input_file.name
        processor.process_nvd_file(input_file, output_file)
        
    else:
        input_dir = args.input_dir or NVD_FILEPATH
        output_dir = args.output_dir or CORPUS_FILEPATH
        
        processed_files = processor.process_directory(input_dir, output_dir)
        print(f"Successfully processed {len(processed_files)} files")

if __name__ == "__main__":
    main()