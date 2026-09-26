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

CampusMate is a Streamlit app that accepts spoken, uploaded, or typed university helpdesk questions. It transcribes speech, classifies the intent with a PyTorch CNN-BiLSTM-attention model trained from scratch, and displays the recognized speech, response, confidence, top-3 intents, and attention-token explanation chips.

## One-command local run

```bash
cd campusmate
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Docker run

```bash
cd campusmate
docker build -t campusmate .
docker run --rm -p 7860:7860 campusmate
```

## Streamlit Community Cloud deployment

1. Go to https://share.streamlit.io and sign in.
2. Click "New app" → choose this repository → set the branch to `main` and the main file path to `app_streamlit.py`.
3. Click "Deploy". The app will install `requirements.txt` and run.

If you want to bake model artifacts into a deployable image, run the training steps locally and commit the `artifacts/` folder before deploying.

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
