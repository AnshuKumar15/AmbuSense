import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Smart Ambulance Risk API")


@app.get("/")
def root():
    return {
        "message": "Smart Ambulance Risk API is running.",
        "docs": "/docs",
        "predict": "/predict"
    }


# Input schema

class VitalsInput(BaseModel):
    heart_rate: list[float]
    spo2: list[float]
    bp_sys: list[float]
    bp_dia: list[float]
    motion: list[float]


# Utility logic (simplified)

def compute_anomaly(hr, spo2):
    hr_slope = np.polyfit(range(len(hr)), hr, 1)[0]
    spo2_slope = np.polyfit(range(len(spo2)), spo2, 1)[0]

    anomaly_score = abs(hr_slope) + abs(spo2_slope)
    anomaly_flag = anomaly_score > 0.5

    return bool(anomaly_flag), float(anomaly_score)


def compute_confidence(motion, spo2):
    conf = 1.0
    if max(motion) > 1.0:
        conf -= 0.4
    if any(np.isnan(spo2)):
        conf -= 0.4
    return max(conf, 0.2)


def compute_risk(hr, spo2, bp_sys, conf):
    risk = 0.0
    if max(hr) > 110:
        risk += 0.3
    if min(spo2) < 92:
        risk += 0.4
    if max(bp_sys) > 140:
        risk += 0.2

    return min(risk * conf * 100, 100)


# API endpoint

@app.post("/predict")
def predict(vitals: VitalsInput):
    hr = np.array(vitals.heart_rate)
    spo2 = np.array(vitals.spo2)
    bp_sys = np.array(vitals.bp_sys)
    motion = np.array(vitals.motion)

    anomaly_flag, _ = compute_anomaly(hr, spo2)
    confidence = compute_confidence(motion, spo2)
    risk_score = compute_risk(hr, spo2, bp_sys, confidence)

    if risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "anomaly": anomaly_flag,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "confidence": round(confidence, 2)
    }
