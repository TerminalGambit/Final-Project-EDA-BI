"""
Configuration Module
====================
Centralized path configuration for the dashboard.
All paths are relative to the project root, making the project portable.
"""

from pathlib import Path

# Project root is the directory containing the app/ folder
# This works regardless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Specific file paths
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.parquet"
METRICS_JSON_PATH = OUTPUTS_DIR / "metrics.json"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
