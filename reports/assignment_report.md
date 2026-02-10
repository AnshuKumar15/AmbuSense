# Smart Ambulance Risk Monitoring System

**AI/ML Engineer Intern Assignment – Gray Mobility**

## 1. Problem Overview

This project is about monitoring patient risk inside a moving ambulance, where vital signs stream every second under noisy, safety-critical conditions. The goal is not perfect prediction accuracy, but early warning, clear reasoning, and controlled alerting in the presence of motion artifacts, missing data, and uncertainty.

The system is built to:

- Separate true physiological deterioration from sensor noise.
- Detect early warning patterns rather than hard threshold breaches.
- Produce explainable risk scores suitable for clinical decision support.
- Operate as a reproducible and deployable ML service.

## 2. Data Generation & Assumptions

### 2.1 Synthetic Time-Series Design

Synthetic multivariate time-series data was generated for 30 minutes (1800 seconds) per patient at 1 Hz, representing continuous ambulance transport. Each timestep includes:

- Heart Rate (HR)
- Oxygen Saturation (SpO₂)
- Blood Pressure (Systolic & Diastolic)
- Motion/Vibration signal

This setup makes it possible to intentionally control:

- Deterioration trajectories
- Sensor artifacts
- Ambulance-specific motion effects

### 2.2 Assumptions

- Physiological changes are gradual, not instantaneous.
- Motion artifacts are intermittent, not continuous.
- SpO₂ is more sensitive to motion and sensor dropout than HR or BP.
- Not all ambulance transports end in critical collapse.

### 2.3 Limitations

- Data is synthetic and does not capture full patient variability.
- No clinician-annotated ground truth labels.
- Motion signal is abstracted rather than device-specific.

These limitations are acknowledged and handled through conservative system design.

## 3. Artifact Detection & Signal Cleaning

Before any anomaly detection, explicit artifact handling was implemented to avoid learning from corrupted signals.

### 3.1 Artifact Types Handled

- Motion-correlated SpO₂ drops
- Sudden HR spikes caused by bumps
- Missing SpO₂ segments due to sensor dropout

Artifacts were detected first, not blindly smoothed.

### 3.2 Cleaning Strategy

- Motion-corrupted SpO₂ values were masked (NaN).
- Short missing segments were linearly interpolated.
- Heart rate was lightly smoothed to remove physiologically impossible jumps.
- Motion signal was intentionally not smoothed, as spikes carry meaning.

Before-and-after plots were generated to verify that:

- Trends were preserved.
- Noise was reduced.
- No artificial stability was introduced.

This step is critical, because downstream ML depends on signal integrity.

## 4. Anomaly Detection (Early Warning)

### 4.1 Design Philosophy

The system avoids per-second anomaly detection. Instead, it looks at short-term trends, reflecting how clinicians reason about deterioration.

### 4.2 Windowing & Features

- Sliding window: 30 seconds
- Step size: 5 seconds

Extracted features:

- HR mean and slope
- SpO₂ mean and slope
- BP systolic variance

These features are interpretable and medically intuitive.

### 4.3 Model Choice

An Isolation Forest was used because it:

- Requires no labels
- Handles multivariate data
- Produces anomaly scores rather than binary decisions

Anomalies represent deviation from normal transport behavior, not immediate danger.

## 5. Risk Scoring & Triage Logic

### 5.1 Risk is Not Binary

Anomaly detection alone does not trigger alerts. Instead, a continuous risk score (0–100) is computed using:

- Anomaly severity
- Vital sign instability
- Signal confidence

### 5.2 Confidence Suppression

Risk is intentionally down-weighted during:

- High motion
- Missing SpO₂ data

This reduces false alerts caused by sensor uncertainty.

### 5.3 Risk Aggregation

Risk aggregation was calibrated to be state-based, not cumulative, which prevents inflation from overlapping anomaly windows.

Risk levels:

- **LOW:** Stable transport
- **MEDIUM:** Concerning trends, monitor closely
- **HIGH:** Sustained deterioration requiring immediate attention

Thresholds were chosen based on achievable score ranges, not arbitrary values.

## 6. Alert Evaluation

### 6.1 Proxy Ground Truth

In the absence of labeled data, deterioration was approximated using sustained vital thresholds (for example, elevated HR with reduced SpO₂). This reflects common early-stage medical AI evaluation practice.

### 6.2 Metrics Reported

- **Precision:** 0.57
- **Recall:** 0.97
- **False Alert Rate:** 0.08
- **Alert Latency:** −145 seconds

### 6.3 Interpretation

- High recall ensures almost no missed deterioration.
- Moderate precision reflects a safety-first design.
- Low false alert rate limits alert fatigue.
- Negative latency indicates early warning, not delayed reaction.

Early alerts are desirable in ambulances, where preparation time matters.

## 7. Failure Analysis

### Failure Case 1 – Motion-Suppressed Deterioration

High motion reduced confidence, delaying escalation despite worsening SpO₂. This is an intentional trade-off to avoid false alarms. Future work could use adaptive confidence weighting.

### Failure Case 2 – False Alert During Transient Instability

Brief motion combined with borderline vitals caused a temporary alert. Sustain logic mitigated alert flickering.

### Failure Case 3 – Delayed Escalation Due to Trend Confirmation

Trend-based detection introduced short delays but reduced noise-driven alerts. This trade-off favors reliability over immediacy.

## 8. Mini ML Service (API)

The final pipeline was exposed through a FastAPI service that:

- Accepts recent vital windows
- Returns:
  - Anomaly flag
  - Risk score
  - Risk level
  - Confidence

The API shows how the system could integrate into a real-time ambulance platform. Interactive documentation is available via `/docs`.

### 8.1 Reproducibility

The repository includes a clear folder structure, a `requirements.txt`, and modular scripts for data generation, preprocessing, anomaly detection, risk scoring, and evaluation. The API service lives in `api/main.py`, and the full pipeline logic is organized under `src/` to support clean training/inference workflows.

## 9. Safety-Critical Considerations

### 9.1 Most Dangerous Failure Mode

False negatives (missed deterioration) are the most dangerous. The system prioritizes recall to minimize this risk.

### 9.2 Reducing False Alerts Without Missing Danger

- Confidence suppression during motion
- Sustained-risk requirements
- Multi-signal agreement before escalation

### 9.3 What Should Never Be Fully Automated

Final clinical decisions must always involve human judgment. This system is designed as decision support, not autonomous diagnosis.

## 10. Conclusion

This project demonstrates a complete, explainable ML pipeline for ambulance risk monitoring, with explicit handling of noisy data, early warning detection, calibrated alerting, and safety-aware evaluation. The design favors engineering judgment over raw accuracy, aligning with real-world medical AI requirements.