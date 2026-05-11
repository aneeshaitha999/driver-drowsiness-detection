# AI-Powered Driver Drowsiness Detection System

Real-time driver monitoring using webcam + MediaPipe Face Mesh to estimate:

- Eye closure (EAR)
- Yawning (MAR)
- Head-down posture (pitch proxy)
- Composite drowsiness risk score

When risk crosses a threshold, the app overlays warning text and triggers an alarm sound.

## Project Structure

```text
driver-drowsiness-detection/
├── app.py
├── detector.py
├── utils.py
├── dashboard.py
├── requirements.txt
├── alarm.wav              # add your own wav file (optional)
├── logs/
├── screenshots/
├── sample_videos/
└── models/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Real-Time Detection

```bash
python app.py --show-fps --mirror
```

Arguments:

- `--camera-index 0` webcam source index
- `--width 960 --height 540` capture size
- `--alarm alarm.wav` alarm file path
- `--log-every-n-frames 5` CSV logging interval

Controls:

- `q` to quit
- `r` to reset blink/yawn counters

## Run Dashboard

```bash
streamlit run dashboard.py
```

Dashboard shows:

- latest risk score
- blink/yawn totals
- alert frequency
- EAR/MAR/risk trends
- recent event table

## Detection Logic Summary

- **EAR**
  - Open: `EAR > 0.25`
  - Closed: `EAR < 0.20`
  - If eyes remain closed for >= 2 seconds -> drowsy signal
- **MAR**
  - Yawn signal: `MAR > 0.60` (with debounce)
- **Head tilt**
  - Downward pitch threshold (configurable)
  - Prolonged head-down contributes to distraction risk
- **Risk score**
  - Eyes closed: +50
  - Yawning: +25
  - Head down: +25
  - Additional minor scoring from blink/yawn accumulation
  - Alert when score >= 70

## Log Output

CSV files are saved in `logs/` with columns:

- timestamp
- ear, mar, pitch_deg
- risk_score
- blink_count, yawn_count
- status
- event

## Notes and Improvements

- Add a real `alarm.wav` for a clear audible alert.
- Tune thresholds per camera and user.
- For production-grade head pose, replace pitch proxy with full PnP pose estimation.
- Extend with distraction detectors (phone use, looking away, seatbelt).

