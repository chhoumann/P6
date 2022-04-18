import pandas as pd
import pathlib
import torch
from data.Dataset import DataDict

from data.lag import series_to_df

LAG = 10


def load_delhi_data() -> DataDict:
    root_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data' / 'delhi_small'
    train_file = data_dir / 'DailyDelhiClimateTrain.csv'
    test_file = data_dir / 'DailyDelhiClimateTest.csv'

    train_series = pd.read_csv(
        train_file,
        parse_dates=["date"],
        index_col="date"
    )["meantemp"]

    test_series = pd.read_csv(
        test_file,
        parse_dates=["date"],
        index_col="date"
    )["meantemp"]

    lagged_train_df = series_to_df(pd.DataFrame(train_series), "meantemp", LAG)
    lagged_test_df = series_to_df(pd.DataFrame(test_series), "meantemp", LAG)

    y_train = lagged_train_df.iloc[:, 0]  # all rows column 0
    # all rows and column 1 and all the following columns
    X_train = lagged_train_df.iloc[:, 1:]

    y_test = lagged_test_df.iloc[:, 0]
    X_test = lagged_test_df.iloc[:, 1:]

    return dict({
        'X_train': torch.from_numpy(X_train.values),
        'y_train': torch.from_numpy(y_train.values),
        'X_test': torch.from_numpy(X_test.values),
        'y_test': torch.from_numpy(y_test.values)
    })
