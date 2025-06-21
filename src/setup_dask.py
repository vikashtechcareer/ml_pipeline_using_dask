# src/setup_dask.py
from dask.distributed import Client

def start_dask():
    client = Client()
    print("Dask client started:")
    print(client.dashboard_link)
    return client
