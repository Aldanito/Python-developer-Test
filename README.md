# Intrusion Detection System

Simple intrusion detection system using YOLO for person detection and OpenCV for visualization.

## Overview

Detects people in video and triggers an alarm when someone enters a restricted zone. You mark the zones beforehand using the zone marker tool.

## Features

- Person detection with YOLOv8
- Interactive zone marking tool
- Alarm triggers on zone intrusion
- Dark video preprocessing (CLAHE)
- FPS and detection count display

## Requirements

- Python 3.11 or newer
- GPU is optional but helps a lot (CUDA for NVIDIA, MPS for Apple Silicon)

## Setup

1. Clone the repo

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. YOLO will auto-download the model on first run (needs internet)

## Usage

### Mark zones first

Run the zone marker:

```bash
python zone_marker.py
```

Controls:

- Left click: add point
- `s`: save current zone
- `r`: reset current zone
- `c`: clear all zones
- `q`: quit and save

Need at least 3 points per zone. Saved to `restricted_zones.json`.

### Run detection

```bash
python main.py
```

Press `q` to exit.

Visual indicators:

- Green: zone boundaries
- Blue: detected people
- Red: people in restricted zones
- "ALARM!" text when intrusion detected
- FPS/detection count in top-left

## Configuration

Edit `config.py` to tweak settings.

Key settings:

- `YOLO_MODEL`: model file (yolov8n.pt = fast, yolov8x.pt = accurate)
- `YOLO_CONFIDENCE`: detection threshold (0.0-1.0, lower = more detections)
- `YOLO_DEVICE`: "cpu", "cuda", or "mps"
- `YOLO_IMGSZ`: input size (320/416/512/640, bigger = slower but better)
- `ALARM_DELAY_SECONDS`: alarm duration after person leaves (default 3s)

Performance tuning:

- `PROCESS_EVERY_N_FRAMES`: skip frames (1 = all, 2 = every other)
- `RESIZE_FACTOR`: downscale before processing (1.0 = full size, 0.75 = 75%)

Accuracy vs speed:

- Best accuracy: `yolov8s.pt` or `yolov8m.pt`, `YOLO_IMGSZ=640`, `PROCESS_EVERY_N_FRAMES=1`
- Best speed: `yolov8n.pt`, `YOLO_IMGSZ=320`, `PROCESS_EVERY_N_FRAMES=2`

## Project files

```
├── main.py              # Main program
├── zone_marker.py       # Tool to mark zones
├── detector.py          # YOLO detection code
├── tracker.py           # Tracking (optional)
├── zone_manager.py      # Load/save zones
├── intrusion_detector.py # Check if person in zone
├── alarm_manager.py     # Handle alarm state
├── config.py            # All settings
├── utils.py             # Helper functions
└── restricted_zones.json # Your marked zones
```

## Zone file format

Zones are saved as JSON. Each zone is a list of [x, y] points:

```json
[
  [[x1, y1], [x2, y2], [x3, y3]],
  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
]
```

## Performance

- Apple Silicon (MPS): 15-30+ FPS

GPU makes a huge difference if available.

## Troubleshooting

**Model won't load:**

- Needs internet for auto-download
- Or download model manually and place in project folder

**Too slow:**

- Use `yolov8n.pt` (smallest model)
- Set `PROCESS_EVERY_N_FRAMES = 2` or 3
- Reduce `YOLO_IMGSZ` to 320
- Use GPU if available

**Not detecting people:**

- Lower `YOLO_CONFIDENCE` (try 0.1)
- Increase `YOLO_IMGSZ` to 640
- Try `yolov8s.pt` or `yolov8m.pt` (more accurate)

**Zones not working:**

- Check `restricted_zones.json` exists
- Run `zone_marker.py` to create zones
- Verify JSON format is valid

## Notes

Built for a coding test assignment.
