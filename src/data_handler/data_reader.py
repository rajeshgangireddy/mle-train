import pandas as pd


class DataReader:
    """
    Simple Data Reader class to read data from a txt or csv file.
    This does not do any preprocessing or cleaning of the data.
    If data has to be read from cloud storages like S3,
    this class can be extended to include that functionality.
    """

    def __init__(
        self, source: str, ignore_columns: list, index_column: int | None = None
    ):
        """
        Initialize the DataReader with the source of the data.

        :param source: Path to the data file. Can be a txt with comma seperated data or a csv file.
        :param ignore_columns: Columns to be ignored while reading the data. Useful when there are columns that are
        not to be processed at all.
        :param index_column: The column with index values. 0 indexed. This column is removed from the data.
        :return: None
        """
        self.data_path = source
        self.ignore_columns = ignore_columns
        self.index_column = index_column

    def _read_data_txt(self) -> pd.DataFrame:
        """
        Read data from a txt file.
        :return: A pandas DataFrame containing the data. No preprocessing is done except for the removing the
        ignore_columns and index_column.
        """
        with open(self.data_path, "r") as file:
            raw_data = file.read()

        # convert to pandas DataFrame
        raw_data = raw_data.split("\n")
        raw_data = [d.split(",") for d in raw_data]
        # Remove any extra quotes from the data
        raw_data = [[d.replace('"', "") for d in data] for data in raw_data]
        header_part = raw_data[0]
        data_part = raw_data[1:]

        # remove ignore_columns columns from the data_part
        if self.ignore_columns:
            data_part = [
                [d for idx, d in enumerate(data) if idx not in self.ignore_columns]
                for data in data_part
            ]

        # remove index column from the data_part
        if self.index_column is not None:
            data_part = [
                [d for idx, d in enumerate(data) if idx != self.index_column]
                for data in data_part
            ]

        return pd.DataFrame(data_part, columns=header_part)

    def _read_data_csv(self) -> pd.DataFrame:
        """
        Read the data from a csv file. NOTE : This is not fully implemented.
        Just a placeholder with minimum code in case the data is from a csv file in the future.
        :return: A pandas DataFrame containing the data.
        """
        return pd.read_csv(self.data_path)

    def read_data(self) -> pd.DataFrame:
        """
        Read data from the source.
        Convert the data to a pandas DataFrame if required.
        :return: DataFrame containing the data
        """
        if self.data_path.endswith(".txt"):
            return self._read_data_txt()
        elif self.data_path.endswith(".csv"):
            return self._read_data_csv()
        else:
            raise ValueError("Only txt and csv files are supported")
