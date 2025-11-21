#!/usr/bin/env python3
"""
Stage 1: File-Level Deduplication

Process each input file independently, removing duplicate FENs within each file.
Keeps the position with the deepest Stockfish evaluation per FEN.
    
This is the first stage of a two-stage deduplication pipeline designed for
memory efficiency when processing large datasets.

Next: Stage 2 will perform global deduplication across all files.
"""
import glob
from pathlib import Path

import polars as pl
from tqdm import tqdm
from loguru import logger


def main():
    """
    Stage 1 - main function
    Deduplicate FENs within a single file (file-level deduplication).
    """
    log_path = Path("logs/data_processing/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "stage_1_deduplication_{time:YYYY-MM-DD}.log",
        rotation="5 MB",
        retention=10
    )
    logger.info("Stage 1: starting process of deduplicating FENs on file level.")

    output_dir = Path("/data/stage_1_deduplicated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1. Get files to process.
    paths = sorted(glob.glob("/data/stage_0_raw/*.parquet"))
    if len(paths) == 0:
        logger.error(
            'Found no files to process in /data/stage_0_raw location. '
            'Please download files using scripts/stage_0_download_data.py script'
        )
        return

    logger.info(f'Found {len(paths)} files to process in /data/stage_0_raw location.')

    for path in tqdm(paths):
        # Step 2. Process every file.
        logger.info(f'Started processing file: {filename}')
        filename = Path(path).name

        q = (
            pl.scan_parquet(path)
            .select(['fen', 'depth', 'cp', 'mate'])
            .sort(['fen', 'depth'], descending=[False, True])
            .unique(subset=['fen'], keep='first')
        )
        logger.success(f'Successfully processed file: {filename}')

        # Step 3. Save file.
        q.sink_parquet(output_dir / filename)
        logger.success(f'File {filename} saved in {output_dir}')

if __name__ == '__main__':
    main()
