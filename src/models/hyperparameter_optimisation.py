import numpy as np
from sklearn.model_selection import RandomizedSearchCV


class HyperparameterTuner:
    """
    Class to perform hyperparameter tuning using RandomizedSearchCV
    """

    def __init__(self, config):
        self.model_params = config["models"]["parameters"]
        self.hpo_params = config["training"]["hyperparameter_optimisation"]
        self.random_state = config["random_seed"]

    def tune(self, model, x_train: np.ndarray, y_train: np.ndarray):
        """
        Perform training with hyperparameter optimisation using RandomizedSearchCV.
        All the parameters required for the search are taken from the config
        :param model: Model to be tuned. Must be supported by sklearn's RandomizedSearchCV
        :param x_train: data points' features
        :param y_train: data points' labels (ground truth)
        :return: Best model after hyperparameter tuning
        """
        search = RandomizedSearchCV(
            model,
            param_distributions=self.model_params,
            n_iter=self.hpo_params["n_iter"],
            scoring=self.hpo_params["scoring"],
            cv=self.hpo_params["cv"],
            random_state=self.random_state,
        )
        search.fit(x_train, y_train)
        return search.best_estimator_
