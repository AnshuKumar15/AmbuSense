# AmbuSense

Smart Ambulance Risk Monitoring System for early warning and explainable risk scoring from streaming vitals in noisy transport conditions.

## Highlights

- Synthetic time-series generation with controlled deterioration and motion artifacts.
- Artifact handling for motion-related noise and missing data.
- Trend-based anomaly detection for early warning.
- Risk scoring with confidence suppression during high uncertainty.
- FastAPI service for real-time integration.

## Project Structure

- [api/main.py](api/main.py): FastAPI service exposing `/predict`.
- [src/data_generation](src/data_generation): Synthetic vitals generation and visualization.
- [src/preprocessing](src/preprocessing): Artifact detection and signal cleaning.
- [src/anomaly_detection](src/anomaly_detection): Trend-based anomaly detection and visualization.
- [src/risk_scoring](src/risk_scoring): Risk scoring logic.
- [src/evaluation](src/evaluation): Alert evaluation metrics.
- [data](data): Raw and processed data outputs.
- [reports/assignment_report.md](reports/assignment_report.md): Full write-up.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive documentation.

### Example Request

```json
{
	"heart_rate": [88, 90, 92, 95, 98, 100],
	"spo2": [98, 97, 96, 95, 94, 93],
	"bp_sys": [120, 122, 125, 128, 130, 132],
	"bp_dia": [80, 82, 83, 84, 85, 86],
	"motion": [0.2, 0.4, 0.3, 0.6, 0.2, 0.1]
}
```

### Example Response

```json
{
	"anomaly": false,
	"risk_score": 28.4,
	"risk_level": "LOW",
	"confidence": 1.0
}
```

## Notes on the Pipeline

- The system prioritizes early warning and robust alerts over peak accuracy.
- Confidence is suppressed during high motion or missing oxygen saturation data.
- Risk levels are state-based to avoid inflation from overlapping windows.

## Report

See the full report in [reports/assignment_report.md](reports/assignment_report.md).