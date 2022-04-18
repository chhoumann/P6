import pandas as pd


def series_to_df(series: pd.DataFrame, key: str, lag=1) -> pd.DataFrame:
    series_n = series.copy()

    for n in range(1, lag + 1):
        series_n[f"lag{n}"] = series_n[key].shift(n)

    series_n = series_n.iloc[lag:]

    return series_n
