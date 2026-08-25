# Image Tampering Detection API + Website

A small forensic tool that flags digitally manipulated images. Built for
the two-part assessment: (1) build a working API around ML/research
logic, (2) integrate that API into a website.

## Architecture

```
             WEBSITE (frontend/index.html)
                    │
                    │ POST /predict  (multipart image upload)
                    ▼
              FastAPI  (backend/main.py)
                    │
                    ▼
      Forensic engine (backend/model/tamper_detect.py)
        ├─ Error Level Analysis (ELA)
        ├─ Noise-inconsistency check
        └─ Edge-artifact check
                    │
                    ▼
      JSON: { prediction, confidence, evidence, ela_image }
                    │
                    ▼
             WEBSITE renders verdict + ELA heatmap
```

## How the detector works

Real camera photos compress fairly uniformly across the whole frame. When
a region has been copy-pasted, spliced, or re-touched, it usually carries
a different compression/noise history than the rest of the image. This
project uses three classical (non-deep-learning) forensic signals to
surface that difference — no training data required, and every result is
explainable rather than a black-box score:

1. **Error Level Analysis (ELA)** — resave the image at a fixed JPEG
   quality and diff it against the original; edited regions "light up".
2. **Noise inconsistency** — real photos have fairly uniform sensor noise;
   spliced regions usually don't match.
3. **Edge artifacts** — pasted regions leave unnatural high-frequency
   edges in the ELA map.

The three scores are combined into one confidence value and a
`Tampered` / `Authentic` verdict.

## Assessment 1 — the API

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- `POST /predict` — upload an image, get back a verdict
- `GET /health` — liveness check
- `GET /docs` — interactive Swagger UI (auto-generated), good for a live
  demo of request/response format

Example:

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:8000/predict
```

```json
{
  "prediction": "Tampered",
  "confidence": 0.77,
  "evidence": {
    "ela_hot_region_pct": 21.6,
    "noise_inconsistency_score": 0.63,
    "edge_artifact_score": 0.03
  },
  "ela_visualization_png_base64": "..."
}
```

## Assessment 2 — the website

```
cd frontend
python3 -m http.server 5500
```

Open `http://127.0.0.1:5500`, make sure the API is running on
`127.0.0.1:8000` (the endpoint is editable at the bottom of the page if
you're running the API somewhere else), then drag in an image. The page
shows the uploaded photo, the verdict, a per-signal evidence breakdown,
and the ELA heatmap side by side.

## Project structure

```
project/
├── backend/
│   ├── main.py              # FastAPI app, /predict and /health routes
│   ├── model/
│   │   └── tamper_detect.py # ELA + noise + edge forensic engine
│   └── requirements.txt
├── frontend/
│   └── index.html           # single-file UI (HTML/CSS/JS, no build step)
├── demo/                    # sample authentic/tampered test images
└── README.md
```

## Notes on calibration

The combine weights and decision threshold in `tamper_detect.py` were
tuned against synthetic authentic/tampered image pairs (see `demo/`,
generated with different noise + JPEG-quality profiles to simulate a
splice). Noise inconsistency turned out to be the strongest signal in
testing, ELA hot-region ratio the second strongest, and edge artifacts
the noisiest (busy natural textures can trigger false positives), hence
the 0.45 / 0.40 / 0.15 weighting. For a production system this would be
replaced or supplemented with a CNN trained on a labeled tampering
dataset (e.g. CASIA) — this classical pipeline was chosen here because it
needs zero training data and every output is explainable, which suits a
fast turnaround demo.
