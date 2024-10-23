import logging

import numpy as np

from .data_reader import DataReader

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class responsible for reading and preparing the data for training the model.
    Any feature engineering or pre-processing of the data should be done here.
    """

    def __init__(self, config):
        self.datetime_features = config["feature_engineering"]["datetime_features"]
        self.categorical_features = config["feature_engineering"][
            "categorical_features"
        ]
        self.target_column = config["feature_engineering"]["target"]

        ignore_columns = (
            config["data"]["ignore_columns"]
            if "ignore_columns" in config["data"]
            else []
        )
        index_column = (
            config["data"]["index_column"] if "index_column" in config["data"] else None
        )

        self.data_reader = DataReader(
            source=config["data"]["path"],
            ignore_columns=ignore_columns,
            index_column=index_column,
        )

    def read_and_prepare_data(self) -> (np.ndarray, np.ndarray):
        """
        Read data from the source (specified in the config). The data is then pre-processed
        (clean, remove any missing values, etc.) and returned as X and y
        :return: X, y as numpy arrays
        """
        df = self._clean_data(self.data_reader.read_data())
        # If any other pre-processing or engineering is required, it can be done here
        x = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # convert x and y to numpy arrays
        x_values = x.values.astype("float32")
        y_values = y.values.astype("float32")

        logger.info(
            f"Loaded data points' features and labels with with shape: {x.shape} and {y.shape}"
        )

        return x_values, y_values

    def _clean_data(self, df):
        df.drop(columns=self.datetime_features, inplace=True)
        df.dropna(inplace=True)
        return df
