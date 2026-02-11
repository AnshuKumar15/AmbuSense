# Smart Ambulance Risk Monitoring System

**AI/ML Engineer Intern Assignment – Gray Mobility**

## 1. Problem Overview

This project focuses on monitoring patient risk inside a moving ambulance, where vital signs stream every second under noisy, safety-critical conditions. The goal is not perfect prediction accuracy, but early warning, explainable reasoning, and controlled alerting in the presence of motion artifacts, missing data, and uncertainty.

The system is designed to:

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
- Motion / Vibration signal

This setup enables intentional control over:

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

These limitations are acknowledged and handled through conservative system design and proxy evaluation.

## 3. Artifact Detection & Signal Cleaning

Before any anomaly detection, explicit artifact handling was implemented to avoid learning from corrupted signals.

### 3.1 Artifact Types Handled

- Motion-correlated SpO₂ drops
- Sudden HR spikes caused by road bumps
- Missing SpO₂ segments due to sensor dropout

Artifacts were detected explicitly, not blindly smoothed.

### 3.2 Cleaning Strategy

- Motion-corrupted SpO₂ values were masked (NaN).
- Short missing segments were linearly interpolated.
- Heart rate was lightly smoothed to remove physiologically impossible jumps.
- Motion signal was intentionally not smoothed, as spikes carry diagnostic meaning.

Before-and-after plots verified that:

- Trends were preserved.
- Noise was reduced.
- No artificial stability was introduced.

This step is critical, as downstream ML is only as reliable as signal integrity.

## 4. Anomaly Detection (Early Warning)

### 4.1 Windowing Strategy (Explicit Explanation)

Instead of detecting anomalies per second, the system uses sliding time windows:

- Window size: 30 seconds
- Step size: 5 seconds

Why windowing is necessary:

- Single-second samples are noisy and unreliable in ambulances.
- Clinicians reason over trends, not individual readings.
- Windowing reduces false positives caused by brief motion spikes or sensor glitches.

Each window summarizes recent patient behavior, enabling early trend detection rather than reactive alerts.

### 4.2 Feature Design (Explicit Explanation)

For each window, the following features were extracted:

- HR mean and slope
- SpO₂ mean and slope
- BP systolic variance

Why these features were chosen:

- Means capture sustained physiological state.
- Slopes capture worsening or improving trends.
- Variance captures instability rather than absolute values.

All features are interpretable and medically intuitive, allowing clinicians to understand why an anomaly was detected.

### 4.3 False Positives in Anomaly Detection

False positives can occur when:

- Short-lived motion spikes resemble deterioration within a window.
- Brief sensor dropouts align with downward SpO₂ trends.

Using overlapping windows and trend-based features reduces sensitivity to single-sample noise, but does not eliminate all transient false positives. These are handled later during risk scoring and confidence suppression.

### 4.4 Model Choice

An Isolation Forest was used because it:

- Requires no labels.
- Handles multivariate data.
- Produces anomaly scores rather than binary decisions.

Anomalies indicate deviation from normal transport behavior, not immediate danger.

## 5. Risk Scoring & Alert Logic

### 5.1 Why Risk Is Not Binary

Anomaly detection alone does not trigger alerts. Instead, a continuous risk score (0–100) is computed using:

- Anomaly severity
- Vital sign instability
- Signal confidence

This prevents alerting on isolated abnormalities.

### 5.2 Why Alerts Trigger or Are Suppressed (Explicit Answer)

Alerts trigger when:

- Multiple signals (HR, SpO₂, BP) show sustained deterioration.
- Risk remains elevated across consecutive windows.
- Confidence in sensor data is sufficiently high.

Alerts are suppressed when:

- Motion is high, reducing signal reliability.
- SpO₂ data is missing or corrupted.
- Abnormality is brief and not sustained.

This ensures the system is conservative when data is unreliable, reducing alert fatigue without ignoring real danger.

### 5.3 Risk Aggregation

Risk aggregation is state-based, not cumulative, preventing inflation due to overlapping windows.

Risk levels:

- **LOW:** Stable transport
- **MEDIUM:** Concerning trends, monitor closely
- **HIGH:** Sustained deterioration requiring immediate attention

Thresholds were chosen based on achievable score ranges, not arbitrary numeric values.

## 6. Alert Evaluation

### 6.1 Proxy Ground Truth

In the absence of clinician-labeled data, deterioration was approximated using sustained vital thresholds (e.g., elevated HR with reduced SpO₂). This reflects common early-stage medical AI evaluation practice.

### 6.2 Metrics Reported

- **Precision:** 0.57
- **Recall:** 0.97
- **False Alert Rate:** 0.08
- **Alert Latency:** −145 seconds

### 6.3 Interpretation (Explicit Error Acceptability)

- High recall (0.97) ensures almost no missed deterioration.
- Moderate precision reflects a safety-first design.
- Low false alert rate limits alert fatigue.
- Negative latency indicates early warning, not delayed reaction.

Acceptable errors in an ambulance:

- False positives (early or unnecessary alerts)
- Slight alert delays due to trend confirmation

Unacceptable errors:

- False negatives (missed deterioration)
- Silent failures during true physiological decline

## 7. Failure Analysis (Task 3B)

### Failure Case 1 – Motion-Suppressed Deterioration

- **What failed:** Alert escalation was delayed despite worsening SpO₂.
- **Why:** High motion reduced confidence and suppressed risk.
- **Improvement:** Adaptive confidence weighting or motion-aware denoising.

### Failure Case 2 – False Alert During Transient Instability

- **What failed:** Temporary alert triggered without sustained danger.
- **Why:** Borderline vitals aligned with a short-lived artifact.
- **Improvement:** Require multi-window confirmation or add a short alert cooldown.

### Failure Case 3 – Delayed Escalation Due to Trend Confirmation

- **What failed:** Escalation occurred later than ideal.
- **Why:** Trend confirmation prioritizes reliability over immediacy.
- **Improvement:** Asymmetric thresholds for faster escalation during sharp multi-signal decline.

## 8. Mini ML Service (API)

The final pipeline was exposed via a FastAPI service that:

- Accepts recent vital windows.
- Returns anomaly flag, risk score, risk level, and confidence.

The API demonstrates how the system could integrate into a real-time ambulance platform. Interactive documentation is available via `/docs`.

## 9. Safety-Critical Considerations (Explicit Answers)

### 9.1 Most Dangerous Failure Mode

The most dangerous failure mode is a false negative: true deterioration is missed or detected too late. In an ambulance, this can lead to delayed interventions, loss of precious preparation time, and worse outcomes. This is more critical than a false positive because clinicians can verify and dismiss extra alerts, but they cannot act on signals that never appear. The system therefore prioritizes recall and uses early-warning trends to surface risk before hard thresholds are crossed.

### 9.2 How to Reduce False Alerts Without Missing Deterioration

- **Confidence suppression during high motion:** down-weights risk when sensors are unreliable, reducing motion-driven false alarms.
- **Sustained-risk requirements:** require risk to remain elevated across consecutive windows to avoid transient spikes.
- **Multi-signal agreement:** escalate only when multiple vitals support a consistent deterioration pattern.
- **Adaptive thresholds:** tighten thresholds during stable transport and loosen slightly when trends consistently worsen.

These mechanisms reduce false alerts while preserving sensitivity to sustained deterioration.

### 9.3 What Should Never Be Fully Automated in Medical AI

Final clinical decisions must never be fully automated. Medical AI should not autonomously diagnose, treat, triage, or override clinician judgment. Humans must remain responsible for verifying alerts, interpreting context (comorbidities, medications, trauma), and deciding interventions. The system is designed as decision support that explains its signals and uncertainty, not as a replacement for clinical reasoning.

## 10. Conclusion

This project presents a complete, explainable ML pipeline for ambulance risk monitoring, emphasizing robustness, early warning, and safety-aware design. The system prioritizes engineering judgment over raw accuracy, aligning with real-world medical AI requirements.