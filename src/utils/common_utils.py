import logging
import os
import time

import yaml


def setup_logger(level: int = logging.INFO) -> None:
    """
    Set up the logger with the specified level
    :param level: logging level
    :return: None
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def read_config_file(config_path: str) -> dict:
    """
    Read the configuration file from the specified path
    :param config_path: path to the configuration file. Must be with at least read permission
    :return: dictionary containing the configuration
    """
    if not config_path.endswith(".yaml"):
        raise ValueError("Only yaml files are supported")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def timestamp_to_string() -> str:
    """
    Return a string with the current timestamp. Can be used for filenames or directory names
    :return: string with the current timestamp
    """
    return time.strftime("%Y%m%d_%H%M%S")
