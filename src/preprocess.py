import os
import requests
import dask.dataframe as dd

def load_and_clean_data(path_pattern="../data/*.parquet"):
    os.makedirs("../data", exist_ok=True)
    local_path = "../data/yellow_tripdata_2023-01.parquet"

    # Download file if not present
    if not os.path.exists(local_path):
        print("Parquet file not found. Downloading...")
        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download completed.")

    # Load parquet files
    df = dd.read_parquet(path_pattern)

    # Basic cleaning
    df = df.dropna()
    df = df[df['trip_distance'] > 0]

    df['pickup_hour'] = dd.to_datetime(df['tpep_pickup_datetime']).dt.hour
    df['is_tip'] = (df['tip_amount'] > 0).astype(int)

    features = ['passenger_count', 'trip_distance', 'pickup_hour']
    df = df[features + ['is_tip']]

    return df
