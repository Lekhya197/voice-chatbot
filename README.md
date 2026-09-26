---
title: CampusMate Voice Helpdesk
emoji: 🎓
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CampusMate - Voice-Enabled Campus Helpdesk Chatbot

Public live URL: `PLACEHOLDER_LIVE_URL`

CampusMate is a FastAPI web app that accepts spoken, uploaded, or typed university helpdesk questions. It transcribes speech, classifies the intent with a PyTorch CNN-BiLSTM-attention model trained from scratch, and displays the recognized speech, response, confidence, top-3 intents, and attention-token explanation chips.

## One-command local run

```bash
cd campusmate
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/build_dataset.py && python training/train.py && python training/evaluate.py
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860`.

## Docker run

```bash
cd campusmate
docker build -t campusmate .
docker run --rm -p 7860:7860 campusmate
```

## Hugging Face Spaces deployment

1. Create a new Hugging Face Space.
2. Choose **Docker** as the SDK and keep the Space public.
3. Clone the Space repo locally.
4. Copy this project into that repo, or push this repository directly to the Space remote.
5. Push to Hugging Face:

```bash
git init
git add .
git commit -m "Deploy CampusMate"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u space main
```

The Docker build runs `python data/build_dataset.py && python training/train.py`, so model artifacts are baked into the image. After build, the Space exposes a public URL; replace `PLACEHOLDER_LIVE_URL` in this README and `REPORT.md`.

## Alternative: Render / Railway

Use the same Dockerfile. Create a new web service from the Git repository, select Docker deployment, expose port `7860`, and set the start command to the Dockerfile default.

## API

- `GET /api/health`
- `POST /api/chat` with `{"text":"what time does the library close"}`
- `POST /api/transcribe` with multipart field `file`

## Tests

```bash
python -m unittest discover -s tests
```

## Final local metrics

The reproducible local training run produced 100.00% test accuracy, 1.000 macro-F1, 1.000 weighted-F1, and 2.23 ms/sample CPU inference latency on the synthetic test split. CNN-only, BiLSTM-only, and hybrid ablations each reached 1.000 macro-F1 on this balanced synthetic dataset.
