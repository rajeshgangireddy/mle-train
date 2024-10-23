from .models import Model, RandomForestModel, XGBModel


class ModelSelector:
    """
    Factory class to get the model based on the model type.
    Only this class is exposed and any model is created only through this class.
    """

    def __init__(self, model_type: str):
        """
        Initialize the ModelSelector with the model type
        :param model_type: a string from the config file specifying the model type
        """
        self.model_type = model_type
        # Mapping of model type to the model class.
        # Maybe this can be better with an enum as the models increase
        self.model_mapping = {
            "XGBClassifier": XGBModel,
            "RandomForest": RandomForestModel,
        }

    def get_model(self) -> Model:
        """
        Return the model class based on the model type
        :return: A model class instance
        """
        model_class = self.model_mapping.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Model type {self.model_type} not supported")
        return model_class()
