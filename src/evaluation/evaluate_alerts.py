from pathlib import Path

import pandas as pd
import numpy as np


def ground_truth(row):
    if (
        row["heart_rate"] > 115 and
        row["spo2"] < 92
    ):
        return 1
    return 0


ROOT_DIR = Path(__file__).resolve().parents[2]
risk_scores_path = ROOT_DIR / "data" / "processed" / "risk_scores.csv"

df = pd.read_csv(risk_scores_path)
df["ground_truth"] = df.apply(ground_truth, axis=1)

df["alert"] = (df["risk_level"] == "HIGH").astype(int)

TP = ((df["alert"] == 1) & (df["ground_truth"] == 1)).sum()
FP = ((df["alert"] == 1) & (df["ground_truth"] == 0)).sum()
FN = ((df["alert"] == 0) & (df["ground_truth"] == 1)).sum()
TN = ((df["alert"] == 0) & (df["ground_truth"] == 0)).sum()

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
false_alert_rate = FP / (FP + TN + 1e-6)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"False Alert Rate: {false_alert_rate:.2f}")

gt_indices = df[df["ground_truth"] == 1].index
alert_indices = df[df["alert"] == 1].index

if len(gt_indices) > 0 and len(alert_indices) > 0:
    latency = alert_indices.min() - gt_indices.min()
    print(f"Alert latency: {latency} seconds")
else:
    print("Latency could not be computed")
