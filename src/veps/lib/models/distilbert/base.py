import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_label_encoders(encoders: Dict[str, Any], path: Path) -> None:
    """Save label encoders to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: dict(v) for k, v in encoders.items()}, f)

def load_label_encoders(path: Path) -> Dict[str, Any]:
    """Load label encoders from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Label encoders not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_json_data(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None