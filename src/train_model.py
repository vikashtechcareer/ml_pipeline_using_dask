from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(df):
    # Features and target
    X = df[["trip_distance", "pickup_hour"]].to_dask_array(lengths=True)
    y = df["is_tip"].to_dask_array(lengths=True)

    # ✅ Fix: Compute chunk sizes for Dask arrays
    X = X.compute_chunk_sizes()
    y = y.compute_chunk_sizes()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test).compute()
    y_test = y_test.compute()
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\n🔍 Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(report)

    # Save model (optional)
    joblib.dump(model, "logistic_model.joblib")

    return model, acc, report
