#!/usr/bin/env python3
"""
Download Lichess chess position evaluations from HuggingFace.

This script downloads parquet files from the Lichess dataset and saves them
to the stage_0_raw directory.

Usage:
    python scripts/download_data.py
"""
from typing import List
import requests
import yaml

from loguru import logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_urls(base_url: str, num_files: int, pattern: str) -> List[tuple]:
    """
    Generate list of (url, filename) tuples for download.
    
    Args:
        base_url: Base URL for the dataset
        num_files: Number of files to download
        pattern: Filename pattern with placeholder for file index
    
    Returns:
        List of (url, filename) tuples
    """
    urls = []
    for i in range(num_files):
        filename = pattern.format(i=i, total=num_files)
        url = f"{base_url}/{filename}"
        urls.append((url, filename))
    return urls


def download_feather_file(url: str, out_path: str):
    """
    Download a Feather file from a URL without loading it into memory,
    save it locally, and return it as a pandas DataFrame.

    Args:
        url: Direct URL to the .feather file.
        out_path: Local destination path for the downloaded file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        OSError: If writing to disk fails.
    """
    logger.info(f"Starting download: {url}")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    logger.info(f"Download finished: {out_path}")


def main():
    """Main data retrieval function"""
    logger.info("Download Lichess chess position evaluations from HuggingFace")
    config = load_config('configs/config.yaml')
    logger.info(config['data']['source']['base_url'])
    # pliki = generate_urls(base_url='https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/data', num_files=16, pattern="train-{i:05d}-of-{total:05d}.parquet")

    # parser.add_argument(
    #     '--config',
    #     type=str,
    #     default='configs/config.yaml',
    #     help='Path to configuration file'
    # )
    # parser.add_argument(
    #     '--log-level',
    #     type=str,
    #     default=None,
    #     choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    #     help='Override logging level from config'
    # )
    # parser
    # args = parser.parse_args()

    logger.debug(f"Przetworzono {1} pozycji")
    logger.warning(f"Pomijam {2} nieprawidłowych FEN-ów")

    logger.info('jakieś info')


# if __name__ == "__main__":
#     main()
main()
