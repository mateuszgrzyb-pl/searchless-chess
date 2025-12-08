"""
Training CNN - S-size.
"""
from pathlib import Path

from loguru import logger

from src.models.shallow_cnn import ShallowCNN


def main():
    """
    Creates and train CNN
    1.18M parameters
    """
    log_path = Path("logs/training/")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(log_path / "training_{time:YYYY-MM-DD}.log",
         rotation="5 MB",
         retention=10)
    logger.info("Creating new shallow CNN.")
    logger.info("1.18M trainable parameters")
    model = ShallowCNN(config={'input_shape':(8, 8, 12)})
    logger.info("Input shape: (8, 8, 12)")
    model.summary()

if __name__ == '__main__':
    main()
