import pandas as pd
from .data_reader import DataReader



class FeatureEngineer:
    def __init__(self, config):
        self.datetime_features = config['feature_engineering']['datetime_features']
        self.categorical_features = config['feature_engineering']['categorical_features']
        self.target_column = config['feature_engineering']['target']

        ignore_columns = config['data']['ignore_columns'] if 'ignore_columns' in config['data'] else []
        index_column = config['data']['index_column'] if 'index_column' in config['data'] else None

        self.data_reader = DataReader(source=config['data']['path'],
                                      ignore_columns=ignore_columns,
                                      index_column=index_column)


    def read_and_prepare_data(self):

        df = self._clean_data(self.data_reader.read_data())
        # If any other pre-processing or engineering is required, it can be done here
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # convert x and y to integers

        x_values = X.values
        y_values = y.values

        x_values = x_values.astype('float32')
        y_values = y_values.astype('float32')

        return x_values, y_values

    def _clean_data(self,df):
        df.drop(columns=self.datetime_features,inplace=True)
        df.dropna(inplace=True)
        return df




