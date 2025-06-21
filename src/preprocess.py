import dask.dataframe as dd
import pandas as pd

def load_and_clean_data(path_pattern="../data/*.parquet"):
    df = dd.read_parquet(path_pattern)

    df = df.dropna()
    df = df[df['trip_distance'] > 0]

    df['pickup_hour'] = dd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['is_tip'] = (df['tip_amount'] > 0).astype(int)

    features = ['passenger_count', 'trip_distance', 'pickup_hour']
    df = df[features + ['is_tip']]

    return df
