#!/usr/bin/env python3
"""
Stage 2: Global Deduplication

Removes duplicate FENs across all files from Stage 1, keeping the position
with the deepest Stockfish evaluation. Produces globally unique positions.

This is the second stage of a two-stage deduplication pipeline. After Stage 1
reduced per-file duplicates, this stage eliminates duplicates that exist
across different files.

Script saves all FENs to single parquet file.
"""
import glob
from pathlib import Path

import polars as pl
from loguru import logger


def main():
    """
    Stage 2 - main function
    Deduplicate FENs globally across all Stage 1 files.
    """
    log_path = Path("logs/data_processing/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "stage_2_deduplication_{time:YYYY-MM-DD}.log",
        rotation="5 MB",
        retention=10
    )
    logger.info("Stage 2: starting process of deduplicating FENs on dataset level.")

    output_dir = Path("data/stage_2_deduplicated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1. Get files to process.
    files = sorted(glob.glob("data/stage_1_deduplicated/*.parquet"))
    if len(files) == 0:
        logger.error(
            'Found no files to process in data/stage_1_deduplicated location. '
            'Please process files using scripts/stage_1_deduplicate_data.py script'
        )
        return

    logger.info(f'Found {len(files)} files to process in data/stage_1_deduplicated location.')

    # Step 2. Process all files at once.
    logger.info('Started processing files from /data/stage_1_deduplicated')
    q = (
        pl.scan_parquet("data/stage_1_deduplicated/*.parquet")
        .select(['fen', 'depth', 'cp', 'mate'])
        .sort(['fen', 'depth'], descending=[False, True])
        .unique(subset=['fen'], keep='first')
    )

    # Step 3. Save file.
    output_file = output_dir / "full_dataset.parquet"
    q.sink_parquet(output_file)
    logger.success('Successfully processed all files.')
    logger.success(f'File saved in {output_dir}')

if __name__ == '__main__':
    main()
