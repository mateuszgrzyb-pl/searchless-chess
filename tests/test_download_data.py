"""Tests for `download_data` module"""
from unittest.mock import patch, MagicMock

from src.utils.tools import load_config
from src.download_data import (
    generate_urls,
    download_parquet_file,
)


def test_load_config(tmp_path):
    """Verify that YAML config loads into dictionary."""
    yaml_content = """
    data:
      source:
        base_url: "https://example.com"
        range_start: 0
        range_stop: 2
        num_of_files: 10
      paths:
        stage_0_raw: "data/raw"
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(str(config_file))
    assert isinstance(config, dict)
    assert config["data"]["source"]["base_url"] == "https://example.com"


def test_generate_urls_basic():
    """Ensure URLs and filenames are generated correctly."""
    base_url = "https://example.com"
    urls = generate_urls(
        base_url=base_url,
        range_start=0,
        range_stop=3,
        num_of_files=100,
        pattern="train-{i:05d}-of-{total:05d}.parquet"
    )

    assert len(urls) == 3
    assert urls[0][0] == "https://example.com/train-00000-of-00100.parquet"
    assert urls[1][1] == "train-00001-of-00100.parquet"


@patch("src.download_data.requests.get")
def test_download_parquet_file(mock_get, tmp_path):
    """Test downloading a parquet file into a path, without real network calls."""
    # Fake response object
    fake_content = b"parquetdata123"
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.headers = {"content-length": str(len(fake_content))}
    fake_resp.iter_content.return_value = [fake_content]
    fake_resp.raise_for_status.return_value = None

    mock_get.return_value = fake_resp

    out_path = tmp_path / "file.parquet"
    download_parquet_file("https://example.com/file.parquet", out_path)

    assert out_path.exists()
    assert out_path.read_bytes() == fake_content
