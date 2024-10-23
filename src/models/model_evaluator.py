from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """
    Class to evaluate a model using various metrics
    """

    @staticmethod
    def evaluate(model, x_test, y_test) -> dict:
        """
        Evaluate the model using various metrics on the provided dataset

        :param model: Any trained model with a predict method.
        :param x_test: Features of the test set data points
        :param y_test: Ground truth labels of the test set data points
        :return: dictionary with evaluation metrics
        """
        y_pred = model.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        return {
            "f1": f1,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

    @staticmethod
    def results_to_pretty_string(results: dict) -> str:
        """
        Convert the results dictionary to a pretty printed string
        :param results: dictionary containing evaluation metrics
        :return: string with pretty printed results
        """
        pretty_results = "\n".join(
            [f"{k.upper().replace('_', ' ')}: {v:.2f}" for k, v in results.items()]
        )
        return "\n" + pretty_results
