#!/usr/bin/env python3
"""
Download Lichess chess position evaluations from HuggingFace.

This script downloads parquet files from the Lichess dataset and saves them
to the stage_0_raw directory.

Usage:
    python scripts/download_data.py
"""
from pathlib import Path
from typing import List

from loguru import logger
import requests
from tqdm import tqdm

from src.utils.tools import load_config

def generate_urls(base_url: str, range_start: int, range_stop: int, num_of_files: int, pattern: str) -> List[tuple]:
    """
    Generate list of (url, filename) tuples for download.
    
    Args:
        base_url: Base URL for the dataset
        range_start: Range start number
        range_stop: Range stop number
        num_of_files: Number of files i repository
        pattern: Filename pattern with placeholder for file index
    
    Returns:
        List of (url, filename) tuples
    """
    urls = []
    for i in range(range_start, range_stop):
        filename = pattern.format(i=i, total=num_of_files)
        url = f"{base_url}/{filename}"
        urls.append((url, filename))
    return urls


def download_parquet_file(url: str, out_path: str):
    """
    Download a Parquet file from a URL without loading it into memory.

    Args:
        url: Direct URL to the .parquet file.
        out_path: Local destination path for the downloaded file, with filename.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        OSError: If writing to disk fails.
    """
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                total_size = int(r.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=out_path.name) as pbar:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        logger.success(f"Saved: {out_path}")
    except requests.HTTPError as e:
        logger.error(f"HTTP error downloading {url}: {e}")
        raise
    except OSError as e:
        logger.error(f"File system error saving to {out_path}: {e}")
        raise

def main():
    """Main data retrieval function"""
    log_path = Path("logs/data_processing/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(log_path / "download_data_{time:YYYY-MM-DD}.log",
         rotation="5 MB",     # New file every 50MB
         retention=10)         # Keep only 10 old files
    logger.info("Download Lichess chess position evaluations from HuggingFace")

    # Step 1. Configuration.
    logger.info('Loading config file.')
    config = load_config('configs/config.yaml')
    base_url = config['data']['source']['base_url']
    range_start = config['data']['source']["range_start"]
    range_stop = config['data']['source']["range_stop"]
    num_of_files = config['data']['source']['num_of_files']
    out_path = config['data']['paths']['stage_0_raw']

    # Step 2. Prepare URLs.
    logger.info('Preparing URLs of the files for download.')
    urls = generate_urls(
        base_url=base_url,
        range_start=range_start,
        range_stop=range_stop,
        num_of_files=num_of_files,
        pattern="train-{i:05d}-of-{total:05d}.parquet")
    logger.debug(f'{len(urls)} URLs prepared')

    # 3. Download files.
    for url, filename in urls:
        logger.debug(f'Downloading file: {filename}')
        out_file = Path(out_path) / filename
        download_parquet_file(url, out_file)

    logger.success('Download completed successfully')

if __name__ == "__main__":
    main()
