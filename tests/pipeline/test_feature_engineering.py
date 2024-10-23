import numpy as np

from src.data_handler.feature_engineering import FeatureEngineer


class TestFeatureEngineering:
    def test_read_and_prepare_data(self):
        config = {
            "feature_engineering": {
                "datetime_features": ["date"],
                "categorical_features": [
                    "Temperature",
                    "Humidity",
                    "Light",
                    "CO2",
                    "HumidityRatio",
                ],
                "target": "Occupancy",
            },
            "data": {
                "path": "tests/data/datatest.txt",
                "ignore_columns": [],
                "index_column": 0,
            },
        }
        feature_engineer = FeatureEngineer(config)
        x, y = feature_engineer.read_and_prepare_data()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape[1] == 5  # 5 features (and should have excluded the date column)
        assert y.shape[0] == 2665  # 2665 data points
