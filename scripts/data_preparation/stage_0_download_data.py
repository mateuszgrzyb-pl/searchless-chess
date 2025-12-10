#!/usr/bin/env python3
"""
Download Lichess chess position evaluations from HuggingFace.

This script downloads parquet files from the Lichess dataset and saves them
to the data/stage_0_raw directory.
"""
from pathlib import Path

from loguru import logger

from src.data_preparation.data_downloading import (
    generate_urls,
    download_parquet_file
)
from src.utils.tools import load_config

def main():
    """Main data retrieval function"""
    log_path = Path("logs/data_processing/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(log_path / "download_data_{time:YYYY-MM-DD}.log",
         rotation="5 MB",     # New file every 5MB
         retention=10)        # Keep only 10 old files
    logger.info("Stage 0: starting the file download process from HuggingFace")

    # Step 1. Configuration.
    logger.info('Loading config file.')
    config = load_config('configs/config.yaml')
    base_url = config['data']['source']['base_data_url']
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
