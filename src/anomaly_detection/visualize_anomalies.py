from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
anomaly_results_path = ROOT_DIR / "data" / "processed" / "anomaly_results.csv"

df = pd.read_csv(anomaly_results_path)

plt.figure(figsize=(10,4))
plt.plot(df["time_sec"], df["anomaly_score"], label="Anomaly Score")
plt.scatter(
    df[df["anomaly_flag"] == 1]["time_sec"],
    df[df["anomaly_flag"] == 1]["anomaly_score"],
    color="red",
    label="Anomaly"
)
plt.legend()
plt.title("Anomaly Scores Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Anomaly Score")
plt.tight_layout()
plt.show()
