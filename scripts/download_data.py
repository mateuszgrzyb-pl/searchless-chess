#!/usr/bin/env python3
"""
Download Lichess chess position evaluations from HuggingFace.

This script downloads parquet files from the Lichess dataset and saves them
to the stage_0_raw directory. It supports:
- Progress tracking with tqdm
- Resume capability (skips already downloaded files)
- Parallel download
- Retry logic for failed downloads
- Validation of downloaded files

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --config configs/config.yaml
"""


def main():
    """Tu pojawi się opis."""

if __name__ == "__main__":
    main()
