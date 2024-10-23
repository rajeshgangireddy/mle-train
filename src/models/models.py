import os
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class Model(ABC):
    """
    Abstract class for creating, saving and loading models.
    Any new model should inherit from this class.
    """

    @abstractmethod
    def create_model(self):
        """
        Abstract method to create a model instance.
        :return:
        """
        pass

    @abstractmethod
    def save_model(self, model, save_dir: str):
        """
        Abstract method to save the model to a file.
        :param model: A trained model
        :param save_dir: A directory with write access to save the model
        :return: None
        """
        pass

    @abstractmethod
    def load_model(self, file_path: str):
        """
        Load an already saved model
        :param file_path: File path (extension specified in the actual implementation)
        :return: the loaded model
        """
        pass


class XGBModel(Model):
    """
    Model class for XGBClassifier.
    """

    def create_model(self) -> XGBClassifier:
        """
        Create an instance of XGBClassifier without any arguments.
        :return:
        """
        return XGBClassifier()

    def save_model(self, model, save_dir: str):
        """
        Save the model to a file.
        :param model: A trained XGBClassifier model
        :param save_dir: directory to which the model should be saved. Must have write permissions.
        :return: None
        """
        file_path = os.path.join(save_dir, "model.json")
        model.save_model(file_path)
        file_path_type = os.path.join(save_dir, "model_type.txt")
        with open(file_path_type, "w") as f:
            f.write("XGBClassifier")

    def load_model(self, file_path: str) -> XGBClassifier:
        """
        Load the XGBClassifier model from a file.
        :param file_path: a json file with the saved model info
        :return: An XGBClassifier model loaded from the filepath
        """
        model = XGBClassifier()
        model.load_model(file_path)
        return model


class RandomForestModel(Model):
    """
    Model class for RandomForestClassifier.
    Only there to show that models can be expanded  if the data scientist wants to add more models.
    Hence, some of the methods are not implemented.
    """

    def create_model(self) -> RandomForestClassifier:
        """
        Create a RandomForestClassifier Model
        :return: A RF Model without any parameter
        """
        return RandomForestClassifier()

    def save_model(self, model, save_dir: str):
        """
        Save the RF model to a file.
        :param model: A trained RF model
        :param save_dir: directory to which the model should be saved. Must have write permissions.
        :return: None
        """
        raise NotImplementedError("Save method not yet implemented for RandomForest")

    def load_model(self, file_path: str):
        """
        Load the RF model from a file.
        :param file_path: a pickle file with the saved model info
        :return: An RF model loaded from the filepath
        """
        raise NotImplementedError("Load method not yet implemented for RandomForest")
