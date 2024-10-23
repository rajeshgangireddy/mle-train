from sklearn.model_selection import train_test_split
from src.data_handler import FeatureEngineer
from src.models import ModelSelector, HyperparameterTuner, ModelEvaluator
import os
class TrainingPipeline:
    """
    Training Pipeline to train (using hyperparameter tuning) and evaluate a model
    """
    def __init__(self, config):
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

    def run(self):
        """
        Run the Training Pipeline. The training_pipeline will:
        1. Load data
        2. Perform Feature Engineering
        3. Split Data
        4. Build and Train Model
        5. Evaluate Model
        """
        random_state = self.config["random_seed"]

        # Step 1 : Read and pre-process data
        X, y = self.feature_engineer.read_and_prepare_data()


        # Step 3: Split Data
        test_size = self.config['training']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        model = self.model_factory_selector.get_model()
        specific_model = model.create_model()

        # Step 4: Hyperparameter tuning
        tuned_model = self.hpo_tuner.tune(specific_model, X_train, y_train)

        # Step 5: Evaluate Model
        evaluation_results = self.evaluator.evaluate(tuned_model, X_test, y_test)
        print(f"Model Evaluation: {evaluation_results}")

        # Step 6: Save Model
        save_dir = self.config["models"]["destination"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_model(tuned_model, save_dir)




