"""
Stage 5: End-to-End Serialization: Target Unification, Tensor Conversion & TFRecord Sharding

This stage transforms raw tabular data directly into TFRecord files ready for GPU acceleration.
It combines target processing, feature extraction, and serialization into a single efficient pass.

This stage:
- Unifies 'mate' and 'cp' evaluations into a single bounded 'score' metric (range [-1, 1])
  using sigmoid scaling (Win Probability), which ensures stable gradients during training.
- Parses 'fen' strings into dense numerical tensors (e.g., 8x8x14 flat arrays).
- Removes the 'depth' column and other unused metadata.
- Shards the dataset into hundreds of smaller files (~100-200MB each) to enable
  parallel reading and perfect shuffling.
- Applies GZIP compression to minimize storage footprint and maximize I/O throughput.

Input:  HuggingFace parquet dataset (10 files, ~31.6M rows each)
Output: data/stage_5_unified/train_*.tfrec (Hundreds of sharded, compressed binary files)
"""
from pathlib import Path

from datasets import load_dataset
from loguru import logger
import polars as pl
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from src.utils.tools import load_config
from src.data_processing import fen_to_tensor


def main():
    """
    Stage 5 - main function
    End-to-End Serialization.
    """
    log_path = Path("logs/data_processing/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "stage_5_unification_{time:YYYY-MM-DD}.log",
        rotation="5 MB",
        retention=10
    )
    logger.info("Stage 5: starting process.")

    # Step 1. Configuration.
    config = load_config('configs/config.yaml')
    data_url = config['data']['source']['processed_data']
    always_white_perspective = config['preprocessing']['encoding']['always_white_perspective']

    options = tf.io.TFRecordOptions(compression_type="GZIP")

    output_dir = Path("data/stage_5_unified")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2. Prepare URLs.
    logger.info('Preparing URLs.')
    data_files = [f"train-0000{i}.parquet" for i in range(0, 10)]

    # Step 3. Process parquets.
    for i, file in enumerate(data_files):
        logger.info(f"Processing file: {file}")
        # preparing dataset from HF
        ds = load_dataset(
            data_url,
            split="train",
            streaming=True,
            data_files=file,
        )

        # streaming file
        for j, batch in enumerate(ds.iter(batch_size=1_000_000)):
            logger.info(f'Streming batch number {j + 1}')
            df = pl.from_dict(batch)

            # Step 3.1. Create new target.
            df = df.with_columns([
                pl.col("fen").str.contains(" w ").alias("color"),
                pl.when(pl.col("mate").is_not_null())
                .then(
                    # Jeśli jest mat:
                    # mate > 0 (dla białego) -> 1.0
                    # mate < 0 (dla czarnego) -> -1.0
                    pl.col("mate").cast(pl.Float32).sign()
                )
                .otherwise(
                    # Jeśli nie ma mata (jest cp): normalizacja tanh
                    (pl.col("cp").cast(pl.Float32) / 1000.0).tanh()
                ).alias("position_eval")
            ])

            # Step 3.2. Adjust perspective to current player.
            # Jeśli ruch ma czarny (color=False), odwracam znak wyniku.
            df = df.with_columns([
                (
                    pl.col("position_eval") * pl.when(pl.col("color")).then(1).otherwise(-1)
                ).alias("position_eval")
            ])

            # Step 3.3. Transform FEN.
            y_data = df["position_eval"].to_numpy()
            fens = df["fen"].to_list()
            x_data = np.array([
                fen_to_tensor(fen, always_white_perspective=always_white_perspective)
                for fen in tqdm(fens, desc='Encoding FEN to tensor.')], dtype=np.uint8)

            # Step 3.4. Write to file.
            with tf.io.TFRecordWriter(str(output_dir / f'file_{i}_{j}.tfrec'), options=options) as writer:
                for board_flat, score_val in tqdm(zip(x_data, y_data), total=len(x_data), desc='Saving'):
                    features = {
                        'board': tf.train.Feature(float_list=tf.train.FloatList(value=board_flat.astype(np.float32).flatten())),
                        'score': tf.train.Feature(float_list=tf.train.FloatList(value=[float(score_val)]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

        logger.info(f'Finished streaming file number {i + 1}')
    logger.success('Serialization completed successfully!')

if __name__ == "__main__":
    main()
