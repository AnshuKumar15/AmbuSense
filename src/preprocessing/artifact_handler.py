from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
raw_data_path = ROOT_DIR / "data" / "raw" / "patient_001.csv"
processed_data_path = ROOT_DIR / "data" / "processed" / "patient_001_cleaned.csv"
before_fig_path = ROOT_DIR / "reports" / "figures" / "before_cleaning.png"
after_fig_path = ROOT_DIR / "reports" / "figures" / "after_cleaning.png"

df = pd.read_csv(raw_data_path)
df_clean = df.copy()

# Heuristics for artifact detection
MOTION_THRESHOLD = 0.9
SPO2_DROP_THRESHOLD = 3

df_clean["spo2_diff"] = df_clean["spo2"].diff()

df_clean["spo2_motion_artifact"] = (
    (df_clean["motion"] > MOTION_THRESHOLD) &
    (df_clean["spo2_diff"] < -SPO2_DROP_THRESHOLD)
)

HR_SPIKE_THRESHOLD = 15

df_clean["hr_diff"] = df_clean["heart_rate"].diff()
df_clean["hr_artifact"] = df_clean["hr_diff"].abs() > HR_SPIKE_THRESHOLD

df_clean["spo2_missing"] = df_clean["spo2"].isna()

df_clean.loc[df_clean["spo2_motion_artifact"], "spo2"] = np.nan

# Fill short gaps in SpO2 readings
df_clean["spo2"] = df_clean["spo2"].interpolate(
    method="linear",
    limit=5
)

# Smooth vitals to reduce single-sample spikes
df_clean["heart_rate"] = (
    df_clean["heart_rate"]
    .rolling(window=5, center=True)
    .median()
)

df_clean["bp_sys"] = (
    df_clean["bp_sys"]
    .rolling(window=5, center=True)
    .mean()
)

# Round to integer-like values while keeping missing SpO2 as NA
df_clean["heart_rate"] = df_clean["heart_rate"].round().astype("Int64")
df_clean["spo2"] = df_clean["spo2"].round().astype("Int64")
df_clean["bp_sys"] = df_clean["bp_sys"].round().astype("Int64")

df[["heart_rate", "spo2", "bp_sys", "motion"]].plot(
    subplots=True,
    figsize=(12, 8),
    title=["HR (Raw)", "SpO₂ (Raw)", "BP Sys (Raw)", "Motion"]
)
plt.tight_layout()
plt.savefig(before_fig_path)
plt.close()

df_clean[["heart_rate", "spo2", "bp_sys", "motion"]].plot(
    subplots=True,
    figsize=(12, 8),
    title=["HR (Cleaned)", "SpO₂ (Cleaned)", "BP Sys (Cleaned)", "Motion"]
)
plt.tight_layout()
plt.savefig(after_fig_path)
plt.close()

df_clean.to_csv(processed_data_path, index=False)

print("Artifact detection and cleaning completed.")
