"""Additional tools"""
import os
from pathlib import Path
from datetime import datetime
import shutil
import yaml

import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def split_parquet_streaming(input_file, output_dir, rows_per_file=100000):
    """
    Split big parquet file into many small files in
    streaming mode using a relatively small amount of RAM. 
    """
    os.makedirs(output_dir, exist_ok=True)
    parquet_file = pq.ParquetFile(input_file)

    part_num = 0
    current_batch = []
    current_rows = 0

    for batch in parquet_file.iter_batches(batch_size=10000):
        current_batch.append(batch)
        current_rows += len(batch)

        if current_rows >= rows_per_file:
            table = pa.Table.from_batches(current_batch)
            output_file = os.path.join(output_dir, f'train-{part_num:05d}.parquet')
            pq.write_table(table, output_file)        
            # Reset
            current_batch = []
            current_rows = 0
            part_num += 1

    # Save data
    if current_batch:
        table = pa.Table.from_batches(current_batch)
        output_file = os.path.join(output_dir, f'train-{part_num:05d}.parquet')
        pq.write_table(table, output_file)


def parse(proto):
    """
    Parse a single serialized TFRecord example containing a chess board tensor and score.

    This function defines the feature schema for each example, where:
      - "board" is an 8×8×12 float32 tensor representing the encoded chess position.
      - "score" is a scalar float32 value representing the evaluation or label.

    Parameters
    ----------
    proto : tf.Tensor (scalar string)
        A serialized TFRecord example.

    Returns
    -------
    tuple(tf.Tensor, tf.Tensor)
        A tuple `(board, score)` where:
          - board : tf.Tensor of shape (8, 8, 12)
          - score : tf.Tensor of shape (1,)
    """
    schema = {
        'board': tf.io.FixedLenFeature([8, 8, 12], tf.float32),
        'score': tf.io.FixedLenFeature([1], tf.float32)
    }
    parsed = tf.io.parse_single_example(proto, schema)
    return parsed['board'], parsed['score']


def create_dataset(file_list, is_training=True, batch_size=8192):
    """
    Create a TensorFlow dataset from a list of TFRecord files containing chess data.

    The dataset pipeline:
      1. Builds a dataset of file paths.
      2. Interleaves multiple TFRecordReaders for high throughput.
      3. Parses each record using `parse`.
      4. Optionally shuffles the dataset during training.
      5. Batches, repeats indefinitely, and prefetches for performance.

    Parameters
    ----------
    file_list : list[str] or tf.Tensor
        List of file paths to TFRecord files, each compressed using GZIP.
    is_training : bool, optional
        If True, the dataset is shuffled to improve training robustness.
    batch_size : int, optional
        Number of samples per batch. Defaults to 8192.

    Returns
    -------
    tf.data.Dataset
        A performant, batched, repeated, and prefetched dataset yielding:
          - board batch of shape (batch_size, 8, 8, 12)
          - score batch of shape (batch_size, 1)
    """
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(50_000)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def setup_training_environment(model_name: str):
    """Creates directories and makes config backup."""
    # 1. Paths
    checkpoints_dir = Path(f"checkpoints/{model_name}/")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    experiment_log_dir = Path(f"logs/training/{model_name}/")
    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = experiment_log_dir / "tensorboard"

    # 2. Config backup
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    shutil.copy('configs/config.yaml', experiment_log_dir / f"config_{current_time}.yaml")

    return experiment_log_dir, checkpoints_dir, tensorboard_dir
