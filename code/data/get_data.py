import enum
from data.Dataset import DataDict

from data.delhi_small.load_data import load_delhi_data


class DataType(enum.Enum):
    """
    Enum for datasets
    """
    delhi_small = 1

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def get_data(dataset: DataType) -> DataDict:
    """
    Loads the data from the given path.
    """
    if dataset == DataType.delhi_small:
        return load_delhi_data()
    else:
        raise ValueError('Invalid dataset')
