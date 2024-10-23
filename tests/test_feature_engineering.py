import pandas as pd
from src.data_handler.feature_engineering import FeatureEngineer

class TestFeatureEngineering:
    def test_read_and_prepare_data(self):
        config = {
            'feature_engineering': {
                'datetime_features': ['date'],
                'categorical_features': ['category'],
                'target': 'target'
            },
            'data': {
                'path': 'tests/data/test_data.csv',
                'ignore_columns': [],
                'index_column': None
            }
        }
        feature_engineer = FeatureEngineer(config)
        X, y = feature_engineer.read_and_prepare_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[1] == 2
        assert y.shape[0] == 3
        assert X.columns.tolist() == ['feature1', 'feature2']
        assert y.tolist() == [1, 2, 3]
