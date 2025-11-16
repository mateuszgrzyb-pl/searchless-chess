"""Loguru setup definition"""
import sys
from pathlib import Path
from loguru import logger

def setup_logger(component: str = "general", level: str = "INFO"):
    """component: 'data_processing', 'training'"""
    log_path = Path("logs") / component
    log_path.mkdir(parents=True, exist_ok=True)

    # Usuń domyślny handler
    logger.remove()

    # Console output - kolorowy, czytelny
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # File output - wszystkie logi
    logger.add(
        log_path / f"{component}_{{time:YYYY-MM-DD}}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        compression="zip"
    )

    # Errors only - osobny plik
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days"
    )
    return logger
