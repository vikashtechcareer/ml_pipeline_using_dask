import time
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import accuracy_score
from dask_ml.linear_model import LogisticRegression as DaskLogistic
from dask_ml.model_selection import train_test_split as dask_train_test_split

def benchmark_sklearn(df):
    start = time.time()

    df_small = df.sample(frac=0.05).compute()  # Downsample for sklearn
    X = df.drop("is_tip", axis=1).to_dask_array(lengths=True)
    y = df["is_tip"].to_dask_array(lengths=True)


    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=0.2)

    model = SkLogistic(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    duration = time.time() - start
    print(f"[Scikit-learn] Accuracy: {acc:.4f}, Time: {duration:.2f}s")

    return {"framework": "scikit-learn", "accuracy": acc, "time": duration}


def benchmark_dask(df):
    start = time.time()

    X = df.drop("is_tip", axis=1).to_dask_array(lengths=True)
    y = df["is_tip"].to_dask_array(lengths=True)


    X_train, X_test, y_train, y_test = dask_train_test_split(X, y, test_size=0.2)
    model = DaskLogistic()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test.compute(), y_pred.compute())

    duration = time.time() - start
    print(f"[Dask-ML] Accuracy: {acc:.4f}, Time: {duration:.2f}s")

    return {"framework": "dask-ml", "accuracy": acc, "time": duration}
