import requests
import zipfile
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime


HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
}


def download_zipfile(year: int, nvd_filepath: Path) -> Optional[Path]:
    url: str = f"https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.json.zip"
    try:
        response: requests.Response = requests.get(url, stream=True, headers=HEADERS)
        response.raise_for_status()
        filepath: Path = nvd_filepath / f"nvdcve-2.0-{year}.json.zip"

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File for {year} downloaded successfully.")
        return filepath
    except requests.ConnectionError as e:
        print(f"Network connection error for {year}: {e}")
    except requests.Timeout as e:
        print(f"Request timed out for {year}: {e}")
    except requests.HTTPError as e:
        print(f"HTTP error occurred for {year}: {e}")
    except (PermissionError, FileNotFoundError) as e:
        print(f"File system error for {year}: {e}")
    except requests.RequestException as e:
        print(f"Unexpected error downloading file for {year}: {e}")
    return None

def extract_nvd_zipfile(zip_filepath: Path, extract_to: Path) -> Optional[Path]:
    """Extract NVD zip file and clean up."""
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        
        try:
            os.remove(zip_filepath)
        except (PermissionError, FileNotFoundError) as e:
            print(f"Error removing zip file {zip_filepath}: {e}")
        
        # Return path to extracted JSON file
        year = zip_filepath.stem.split('-')[-1]
        json_filepath = extract_to / f"nvdcve-2.0-{year}.json"
        print(f"Data extracted successfully: {json_filepath}")
        return json_filepath
        
    except zipfile.BadZipFile as e:
        print(f"Invalid or corrupted zip file {zip_filepath}: {e}")
    except (PermissionError, FileNotFoundError, zipfile.LargeZipFile) as e:
        print(f"File system error with {zip_filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error extracting {zip_filepath}: {e}")
    
    return None

def download_nvd_data_range(output_dir: Path, start_year: int = 2002, end_year: int = 2026) -> List[Path]:
    """Download and extract NVD CVE data for a range of years."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    
    for year in range(start_year, end_year):
        zip_filepath = download_zipfile(year, output_dir)
        if zip_filepath:
            json_filepath = extract_nvd_zipfile(zip_filepath, output_dir)
            if json_filepath:
                extracted_files.append(json_filepath)
    
    return extracted_files

def download_latest_nvd_data(output_dir: Path) -> List[Path]:
    """Download the most recent NVD data (last 2 years)."""
    current_year = datetime.now().year
    return download_nvd_data_range(output_dir, current_year - 1, current_year + 1)


def main(args):
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