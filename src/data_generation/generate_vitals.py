import numpy as np
import pandas as pd


np.random.seed(42)

DURATION_SEC = 30 * 60  # 30 minutes
time = np.arange(DURATION_SEC)


# Heart Rate (HR)
hr = np.random.normal(75, 3, DURATION_SEC)

# Gradual deterioration after 10 mins
for i in range(600, DURATION_SEC):
    hr[i] += (i - 600) * 0.04


# SpO2
spo2 = np.random.normal(98, 0.3, DURATION_SEC)

for i in range(600, DURATION_SEC):
    spo2[i] -= (i - 600) * 0.011

spo2 = np.clip(spo2, 85, 100)


# Blood Pressure
bp_sys = np.random.normal(120, 4, DURATION_SEC)
bp_dia = np.random.normal(80, 3, DURATION_SEC)

# Add instability later
bp_sys[1200:] += np.random.normal(0, 8, DURATION_SEC - 1200)
bp_dia[1200:] += np.random.normal(0, 3, DURATION_SEC - 1200)


# Motion Signal
motion = np.random.normal(0.2, 0.05, DURATION_SEC)

# Ambulance bumps
for i in range(0, DURATION_SEC, 120):
    motion[i:i + 5] += np.random.uniform(0.8, 1.2)


# Motion-induced artifacts
artifact_idx = motion > 0.9
spo2[artifact_idx] -= np.random.uniform(3, 6)

# Missing data simulation
missing_idx = np.random.choice(DURATION_SEC, 40, replace=False)
spo2[missing_idx] = np.nan


# Convert vitals to integers (preserve missing SpO2 values)
hr = np.rint(hr).astype(int)
bp_sys = np.rint(bp_sys).astype(int)
bp_dia = np.rint(bp_dia).astype(int)
spo2 = pd.Series(spo2).round().astype("Int64")


# Create DataFrame
df = pd.DataFrame(
    {
        "time_sec": time,
        "heart_rate": hr,
        "spo2": spo2,
        "bp_sys": bp_sys,
        "bp_dia": bp_dia,
        "motion": motion,
    }
)

df.to_csv("data/raw/patient_001.csv", index=False)
print("Synthetic patient data generated!")
