import logging

from src.training_pipeline import TrainingPipeline
from src.utils import read_config_file, setup_logger

logger = logging.getLogger(__name__)


def main():
    """
    Create and run the training pipeline. For the demo I am using a fixed path config file
    which has all the parameters required to run the training pipeline. The config file also has fixed directories
    to read data from and save models to.
    If not for the demo, ideally, the config file would be made as an optional argument to the main function.
    :return: None
    """
    setup_logger(level=logging.INFO)
    demo_config_path = "src/configs/config.yaml"
    pipeline_config = read_config_file(demo_config_path)
    logger.info("Creating Training Pipeline from configuration")
    pipeline = TrainingPipeline(pipeline_config)
    pipeline.run()


if __name__ == "__main__":
    main()
    #
