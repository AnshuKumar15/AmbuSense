import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/patient_001.csv")

df[[
    "heart_rate",
    "spo2",
    "bp_sys",
    "bp_dia",
    "motion"
]].plot(
    subplots=True,
    figsize=(12, 8),
    title=[
        "Heart Rate",
        "SpO₂",
        "Blood Pressure (Systolic)",
        "Blood Pressure (Diastolic)",
        "Motion"
    ]
)

plt.tight_layout()
plt.show()
