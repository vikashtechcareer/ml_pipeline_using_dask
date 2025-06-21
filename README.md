
# 🧠 Distributed ML Pipeline using Dask

This Mini Project repository demonstrates a modular, scalable Machine Learning pipeline using **Dask** for distributed data processing. The pipeline is designed to handle NYC Taxi trip data, and includes preprocessing, model training, performance evaluation, and visualization.

---

## 📌 Key Features

- ✅ Efficient processing of large datasets using Dask
- ✅ Modular Python scripts and corresponding Jupyter Notebooks
- ✅ Logistic Regression model for classification
- ✅ Visualization of model evaluation metrics
- ✅ Trained model export and result reproducibility

---

## 📁 Project Structure

```
ml_pipeline_using_dask/
├── data/
│   └── yellow_tripdata_2023-01.parquet       # NYC Taxi dataset (Parquet format)
│
├── notebooks/
│   ├── 01_preprocess.ipynb                   # Data preprocessing
│   ├── 02_train_model.ipynb                  # Model training using Logistic Regression
│   ├── 03_performance_comparison.ipynb       # Evaluation and metrics generation
│   ├── 04_visualize_metrics.ipynb            # Plotting confusion matrix
│   ├── logistic_model.joblib                 # Serialized trained model
│   └── visualizations/
│       └── confusion_matrix.png              # Confusion matrix output
│
├── src/
│   ├── preprocess.py                         # Preprocessing logic
│   ├── train_model.py                        # Model training logic
│   ├── performance_compare.py                # Metric calculations (precision, recall, etc.)
│   ├── visualize.py                          # Confusion matrix plotting
│   └── setup_dask.py                         # Dask cluster setup
│
├── requirements.txt                          # Python dependencies
└── README.md                                 # This documentation
```

---

## 🚀 How to Run This Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vikashtechcareer/ml_pipeline_using_dask.git
cd ml_pipeline_using_dask
```

### 2️⃣ Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Dask Scheduler and Workers

```bash
python src/setup_dask.py
```

This script sets up a local Dask distributed cluster for parallel processing.

---

## 🔄 Pipeline Steps

### 🧹 Data Preprocessing (`src/preprocess.py`)

- Loads the Parquet dataset
- Converts pickup & dropoff timestamps
- Calculates trip duration and filters invalid entries
- Creates new features (e.g., hour of day, day of week)
- Stores cleaned dataset as `processed_data.csv`

### 🤖 Model Training (`src/train_model.py`)

- Reads the preprocessed CSV
- Converts categorical features using `LabelEncoder`
- Splits into training and testing sets
- Trains a `LogisticRegression` model using Dask-ML
- Saves model to `logistic_model.joblib`

### 📊 Performance Evaluation (`src/performance_compare.py`)

- Loads trained model and test data
- Computes `precision`, `recall`, `f1-score`, and `accuracy`
- Stores metrics in a dictionary and prints summary

### 📈 Visualization (`src/visualize.py`)

- Loads predictions and actual values
- Generates a confusion matrix plot using `matplotlib`
- Saves output to `visualizations/confusion_matrix.png`

---

## 📘 Notebooks

Each script has a corresponding notebook under `notebooks/` for interactive development and debugging:
- `01_preprocess.ipynb`
- `02_train_model.ipynb`
- `03_performance_comparison.ipynb`
- `04_visualize_metrics.ipynb`

---

## 💾 Outputs

- `processed_data.csv` — Cleaned dataset
- `logistic_model.joblib` — Saved model
- `confusion_matrix.png` — Evaluation visualization
- `metrics.json` or inline output — Performance results

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [Dask](https://www.dask.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
