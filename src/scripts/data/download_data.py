from pathlib import Path
import argparse
from datetime import datetime

from veps.lib.data.download_nvd import download_nvd_data_range, download_latest_nvd_data
from veps.config import NVD_FILEPATH


def main():
    parser = argparse.ArgumentParser(description="Download NVD CVE data")
    parser.add_argument("--years", type=str, help="Year range (e.g., '2020-2024')")
    parser.add_argument("--latest", action="store_true", help="Download latest data only")
    parser.add_argument("--output-dir", type=str, default=NVD_FILEPATH, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.latest:
        print("Downloading latest NVD data...")
        files = download_latest_nvd_data(output_dir)
    elif args.years:
        start, end = map(int, args.years.split('-'))
        print(f"Downloading NVD data for years {start}-{end}")
        files = download_nvd_data_range(output_dir, start, end + 1)
    else:
        print("Downloading all available NVD data...")
        current_year = datetime.now().year
        files = download_nvd_data_range(output_dir, end_year=current_year)
    
    print(f"\nCompleted: Downloaded {len(files)} files to {output_dir}")

if __name__ == "__main__":
    main()