from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

ROOT_DIR = Path(__file__).resolve().parents[2]
cleaned_data_path = ROOT_DIR / "data" / "processed" / "patient_001_cleaned.csv"
output_path = ROOT_DIR / "data" / "processed" / "anomaly_results.csv"

df = pd.read_csv(cleaned_data_path)

signals = ["heart_rate", "spo2", "bp_sys"]

WINDOW_SIZE = 30
STEP_SIZE = 5

features = []
timestamps = []

for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
    window = df.iloc[start:start + WINDOW_SIZE]

    feature_vector = {
        "hr_mean": window["heart_rate"].mean(),
        "hr_slope": np.polyfit(range(WINDOW_SIZE), window["heart_rate"], 1)[0],
        "spo2_mean": window["spo2"].mean(),
        "spo2_slope": np.polyfit(range(WINDOW_SIZE), window["spo2"], 1)[0],
        "bp_sys_var": window["bp_sys"].var()
    }

    features.append(feature_vector)
    timestamps.append(start)

X = pd.DataFrame(features)
X_features = X.copy()

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X_features)

X["anomaly_score"] = model.decision_function(X_features)
X["anomaly_flag"] = model.predict(X_features)

X["anomaly_flag"] = (X["anomaly_flag"] == -1).astype(int)
X["time_sec"] = timestamps

X.to_csv(output_path, index=False)
