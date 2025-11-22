#!/usr/bin/env python3
"""
Stage 3: Split for distribution

Splits the globally deduplicated dataset from Stage 2 into 10 equal parts
for efficient distribution and incremental loading.

This stage prepares the final dataset for publication on HuggingFace, making
it accessible to the ML community while enabling users to download and process
data in manageable chunks.

Input:  data/stage_2_deduplicated/full_dataset.parquet (~316M rows, single file)
Output: data/stage_3_split/*.parquet (10 files, ~31.6M rows each)

The split files are named train-00000.parquet through train-00009.parquet,
following HuggingFace dataset conventions for automatic integration.
"""
from pathlib import Path

from loguru import logger

from src.utils.tools import split_parquet_streaming

def main():
    """
    Stage 3 - main function
    Split big parquet file into 10 smaller chunks.
    """
    log_path = Path("logs/data_processing/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "stage_3_splitting_{time:YYYY-MM-DD}.log",
        rotation="5 MB",
        retention=10
    )
    logger.info("Stage 3: starting process of splitting parquet file before distributing to HuggingFace.")

    output_dir = Path("data/stage_3_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1. Get file to process.
    file = Path("data/stage_2_deduplicated/full_dataset.parquet")

    if not file.exists():
        logger.error(
            'Found no file to process in data/stage_2_deduplicated location. '
            'Please process files using scripts/stage_2_deduplicate_data.py script'
        )
        return

    logger.info('Found file to process in data/stage_2_deduplicated')

    # Step 2. Splitting file into smaller chunks.
    logger.info('Started processing "full_dataset.parquet" file')
    split_parquet_streaming(
        input_file=file,
        output_dir=output_dir,
        rows_per_file=1_000_000)

    logger.success('Successfully processed full_dataset_parquet')
    logger.success(f'Files saved in {output_dir}')

if __name__ == '__main__':
    main()
