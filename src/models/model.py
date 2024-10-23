import os

from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os

class Model(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def save_model(self, model, file_path: str):
        pass

    @abstractmethod
    def load_model(self, file_path: str):
        pass


class XGBModel(Model):
    def create_model(self):
        return XGBClassifier()

    def save_model(self, model, save_dir: str):
        file_path = os.path.join(save_dir, "model.json")
        model.save_model(file_path)
        file_path_type = os.path.join(save_dir, "model_type.txt")
        with open(file_path_type, "w") as f:
            f.write("XGBClassifier")

    def load_model(self, file_path: str):
        model = XGBClassifier()
        model.load_model(file_path)
        return model


class RandomForestModel(Model):
    def create_model(self):
        return RandomForestClassifier()

    def save_model(self, model, file_path: str):
        raise NotImplementedError("Save method not yet implemented for RandomForest")

    def load_model(self, file_path: str):
        raise NotImplementedError("Load method not yet implemented for RandomForest")

class ModelSelector:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_model(self):
        if self.model_type == "XGBClassifier":
            return XGBModel()
        elif self.model_type == "RandomForest":
            return RandomForestModel()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

