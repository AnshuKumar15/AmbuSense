from pathlib import Path

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
cleaned_data_path = ROOT_DIR / "data" / "processed" / "patient_001_cleaned.csv"
anomaly_data_path = ROOT_DIR / "data" / "processed" / "anomaly_results.csv"
output_path = ROOT_DIR / "data" / "processed" / "risk_scores.csv"

df_clean = pd.read_csv(cleaned_data_path)
df_anom = pd.read_csv(anomaly_data_path)

df_clean["risk_score"] = 0.0
df_clean["risk_confidence"] = 1.0

anom_min = df_anom["anomaly_score"].min()
anom_max = df_anom["anomaly_score"].max()

df_anom["anom_risk"] = (
    (anom_max - df_anom["anomaly_score"]) /
    (anom_max - anom_min)
)

def vital_risk(row):
    risk = 0

    if row["heart_rate"] > 110:
        risk += 0.3
    if row["spo2"] < 92:
        risk += 0.4
    if row["bp_sys"] > 140:
        risk += 0.2

    return min(risk, 1.0)

def confidence(row):
    conf = 1.0

    if row["motion"] > 1.0:
        conf -= 0.4
    if pd.isna(row["spo2"]):
        conf -= 0.4

    return max(conf, 0.2)


WINDOW_SIZE = 30

for _, w in df_anom.iterrows():
    start = int(w["time_sec"])
    end = start + WINDOW_SIZE

    df_clean.loc[start:end, "risk_score"] = np.maximum(
    df_clean.loc[start:end, "risk_score"],
    0.5 * w["anom_risk"]
)


df_clean["vital_risk"] = df_clean.apply(vital_risk, axis=1)

df_clean["risk_score"] = np.maximum(
    df_clean["risk_score"],
    df_clean["vital_risk"]
)


df_clean["risk_confidence"] = df_clean.apply(confidence, axis=1)
df_clean["risk_score"] = (
    df_clean["risk_score"] * df_clean["risk_confidence"]
)

df_clean["risk_score"] = (
    df_clean["risk_score"] * df_clean["risk_confidence"] * 100
).clip(0, 100)


def risk_level(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"

df_clean["risk_level"] = df_clean["risk_score"].apply(risk_level)

df_clean.to_csv(output_path, index=False)
print("Risk scoring completed.")

