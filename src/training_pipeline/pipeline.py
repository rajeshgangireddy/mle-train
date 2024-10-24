import logging
import os

import yaml
from sklearn.model_selection import train_test_split

from src.data_handler import FeatureEngineer
from src.models import HyperparameterTuner, ModelEvaluator, ModelSelector
from src.utils import timestamp_to_string

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Training Pipeline to train (using hyperparameter tuning) and evaluate a model
    """

    def __init__(self, config) -> None:
        """
        Initialize the Training Pipeline
        :param config: Configuration dictionary with all the necessary parameters
        """
        self.config = config
        model_type = config["models"]["type"]

        self.feature_engineer = FeatureEngineer(config=config)
        self.model_factory_selector = ModelSelector(model_type)
        self.hpo_tuner = HyperparameterTuner(config=config)
        self.evaluator = ModelEvaluator()

    def run(self) -> None:
        """
        Run the Training Pipeline. The training_pipeline will:
        1. Load and pre-process data
        2. Split Data into train and test split
        3. Build a model
        4. Perform training using hyperparameter optimisation
        5. Evaluate the best model on the test set
        6. Save Model
        """
        random_state = self.config["random_seed"]

        # Step 1 : Read and pre-process data
        logger.info("Reading and Pre-processing Data")
        X, y = self.feature_engineer.read_and_prepare_data()

        # Step 2: Split Data into train and test split.

        # The train split will be split automatically for training and validation during the
        # cross-validation
        test_size = self.config["training"]["test_size"]
        logger.info(f"Splitting data into train and test with test size: {test_size}")
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Step 3 : Build the model
        logger.info("Building Model")
        model = self.model_factory_selector.get_model()
        specific_model = model.create_model()

        # Step 4: Training with Hyperparameter optimisation.

        logger.info(
            "Training Model with Hyperparameter Optimisation. Note: This may take a while"
        )
        best_performance_model = self.hpo_tuner.tune(specific_model, x_train, y_train)

        # Step 5: Evaluate Model
        logger.info("Evaluating Model on Test Set")
        evaluation_results = self.evaluator.evaluate(
            best_performance_model, x_test, y_test
        )
        logger.info(
            f"Model Evaluation: {self.evaluator.results_to_pretty_string(evaluation_results)}"
        )

        # Step 6: Save Model and a copy of the config file
        save_dir = os.path.join(
            self.config["models"]["destination"], timestamp_to_string()
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_model(best_performance_model, save_dir)
        logger.info(f"Model saved at: {save_dir}")

        # save a copy of config file - Good to have this information for reproducibility
        # or selecting a particular model in future
        with open(os.path.join(save_dir, "config_used_for_training.yaml"), "w") as file:
            yaml.dump(self.config, file)
