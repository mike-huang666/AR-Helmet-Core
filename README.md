# AR Helmet HUD: Real-Time Collision Warning Prototype

A small proof-of-concept that simulates the heads-up display (HUD) idea from our
university business project, *"AR Safety Helmet: Guardian of the Future"*
(Fu Jen Catholic University, Business Management program, 2026).

The business plan envisions a motorcycle helmet visor that shows hazard
warnings without the rider looking down at a phone. This repository contains
only a **webcam-based software demo** of that warning overlay. It is not the
product, and it does not run on helmet hardware.

## What it does

1. **Category filtering** – runs YOLOv8n (pretrained COCO weights) and keeps only
   road-relevant classes: person, car, motorcycle, bus, truck, traffic light
   (plus "cell phone" so it can be tested indoors).
2. **Proximity-based risk levels** – uses the fraction of the frame covered by
   each detection's bounding box as a rough proximity cue:
   - > 20% of frame: red, `DANGER: TOO CLOSE!`
   - > 8% of frame: yellow, `CAUTION`
   - otherwise: green, `SAFE`
3. **HUD-style overlay** – draws boxes, labels, a status line and a center crosshair
   on the live camera feed.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ar_helmet_hud.py
```

Press `q` in the video window to quit. The first run downloads `yolov8n.pt`.

> The camera is opened with `cv2.CAP_AVFOUNDATION` (macOS). On Windows or Linux,
> change that line to `cv2.VideoCapture(0)`.

## Limitations (please read)

- Bounding-box area is **not** real distance. A bus far away and a phone held close
  can produce the same ratio. Thresholds (0.20 / 0.08) are hand-picked, not
  calibrated or validated against measured distances.
- Single webcam, no radar or depth sensor, no motion or approach-speed estimate.
- Not tested on a moving vehicle, in rain, at night, or on any helmet hardware.
- Not a safety device. Do not rely on it for real riding.

## How it was built

- An extra proof-of-concept I built on my own, separate from the team's business plan
  (my role in the team plan was hardware and sensor integration).
- The code was written with the help of Google Gemini (AI assistant), then run and tested by me.
- Detection model: Ultralytics YOLOv8n.

## License

Code: MIT (see `LICENSE`).
Note: Ultralytics YOLO is released under AGPL-3.0 (or a commercial license);
check their current terms before distributing anything built on it.
