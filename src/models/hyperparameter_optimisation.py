from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter tuning class
class HyperparameterTuner:
    def __init__(self, config):
        model_params = config["models"]["parameters"]
        hpo_params = config["training"]["hyperparameter_optimisation"]
        self.model_params = model_params
        self.hpo_params = hpo_params
        self.random_state = config["random_seed"]

    def tune(self, model, X_train, y_train):
        search = RandomizedSearchCV(
            model, param_distributions=self.model_params,
            n_iter=self.hpo_params["n_iter"], scoring=self.hpo_params["scoring"],
            cv=self.hpo_params["cv"], random_state=self.random_state
        )
        search.fit(X_train, y_train)
        return search.best_estimator_