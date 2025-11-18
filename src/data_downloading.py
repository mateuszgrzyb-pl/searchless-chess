"""
Functions needed to retrieve data with Lichess positions from HuggingFace
"""
from typing import List

from loguru import logger

import requests
from tqdm import tqdm


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
