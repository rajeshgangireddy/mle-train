import os
import yaml
from src.training_pipeline import TrainingPipeline

def load_config(config_path: str) -> dict:
    """
    Load configuration file from the specified config_path
    :param config_path: path to the configuration file. Must be in
        yaml format and with read permissions.
    :return: dictionary containing the configuration
    """
    if not config_path.endswith('.yaml'):
        raise ValueError("Only yaml files are supported")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File not found at: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    pipeline_config = load_config("src/configs/config.yaml")
    pipeline = TrainingPipeline(pipeline_config)
    pipeline.run()
