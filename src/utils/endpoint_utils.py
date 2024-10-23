import numpy as np


def data_to_features(data: list | dict, feature_order) -> np.ndarray:
    """
    Convert JSON data to features using the feature engineer
    :param data: JSON data
    :param feature_order: Order of features in which the data should be returned
    :return: Features as a numpy array
    """
    valid_feature_names = feature_order
    if isinstance(data, list):
        features = []
        for data_point in data:
            features.append(json_data_item_to_features(data_point, valid_feature_names))
        return np.array(features)
    elif isinstance(data, dict):
        return np.array([json_data_item_to_features(data, valid_feature_names)])
    else:
        raise ValueError("Invalid data format. Please provide a list or dictionary")


def json_data_item_to_features(data_point: dict, feature_order: list) -> list:
    """
    Convert a single JSON data item to features using the feature engineer
    :param data_point: A single data item
    :param feature_order: Order of features in which the data should be returned
    :return: Features
    """
    feature = []
    for feature_name in feature_order:
        if feature_name not in data_point:
            raise ValueError(f"Feature {feature_name} not found in data")
        feature.append(data_point[feature_name])
    return feature
