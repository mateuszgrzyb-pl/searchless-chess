"""Additional tools"""
import os
import yaml
import pyarrow as pa
import pyarrow.parquet as pq


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
            output_file = os.path.join(output_dir, f'part_{part_num:03d}.parquet')
            pq.write_table(table, output_file)        
            # Reset
            current_batch = []
            current_rows = 0
            part_num += 1

    # Save data
    if current_batch:
        table = pa.Table.from_batches(current_batch)
        output_file = os.path.join(output_dir, f'part_{part_num:03d}.parquet')
        pq.write_table(table, output_file)
        print(f"Zapisano {output_file} ({len(table)} wierszy)")
