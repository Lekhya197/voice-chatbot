# CampusMate - Voice-Enabled Campus Helpdesk Chatbot Report

Live URL: `PLACEHOLDER_LIVE_URL`

## Abstract

CampusMate is a closed-domain, voice-enabled campus helpdesk chatbot for university support questions. The system accepts browser speech recognition, server-side uploaded audio transcription, or plain typed input. It converts the user query into text, preprocesses it, classifies it with a deep-learning intent model trained from scratch, and returns an intent-specific response. The frontend explicitly displays the recognized speech text and the chatbot response, along with confidence, top-3 intent probabilities, and a small attention-token explanation. The backend is implemented with FastAPI and the model is implemented in PyTorch. The project is packaged with Docker for Hugging Face Spaces deployment on the free CPU tier.

## Problem Statement

University students often ask repeated operational questions about admissions, fees, hostels, library access, exams, attendance, Wi-Fi, buses, placements, and grievances. A lightweight helpdesk chatbot can reduce repetitive administrative load while giving students fast answers. The academic objective of this project is to demonstrate an end-to-end speech-to-intent pipeline without depending on a hosted LLM or paid classifier API. CampusMate therefore trains a supervised neural intent classifier from a locally created dataset and combines it with two speech-recognition paths: the Web Speech API for low-latency browser transcription and faster-whisper tiny.en for server-side fallback.

## System Architecture

```text
User voice / typed text
        |
        v
Browser Web Speech API OR uploaded audio -> faster-whisper tiny.en
        |
        v
Recognized speech text
        |
        v
Preprocessing: lowercase, punctuation stripping, whitespace tokenization, padding
        |
        v
CNN-BiLSTM-Attention PyTorch classifier
        |
        v
Intent + confidence + top-3 probabilities + attention tokens
        |
        v
Template response or fallback response
        |
        v
FastAPI JSON response -> vanilla HTML/CSS/JS interface
```

## Dataset

The dataset is created locally in `data/intents.json`; no external data is downloaded. It contains 22 university helpdesk intents: greeting, goodbye, thanks, bot_identity, admission_process, admission_eligibility, fee_payment, fee_structure, scholarship_info, hostel_allotment, hostel_rules, mess_menu, library_timings, library_borrowing, exam_schedule, exam_results, attendance_policy, course_registration, placement_info, campus_wifi_support, transport_bus, and grievance_redressal. Each intent has at least 30 natural utterances and 4-6 response templates. The utterances include polite forms, short commands, filler words, contractions where useful, and mildly misspelled speech-recognition-style variants such as "libary" and "reslts".

`data/build_dataset.py` deterministically flattens the JSON into `data/dataset.csv` with columns `text,label`, using seed 42. The generated class distribution is balanced: each of the 22 intents has 34 utterances, for a total of 748 samples. The training script uses a stratified 70/15/15 train/validation/test split with seed 42. Because the dataset is synthetic, it is clean and balanced, which helps model performance but overstates real-world robustness. Real campus logs would include more ambiguity, accents, mixed languages, incomplete sentences, and out-of-scope requests.

| Intent | Samples |
|---|---:|
| Each of 22 intents | 34 |
| Total | 748 |

## Methodology

Text is lowercased, punctuation is removed except apostrophes, repeated whitespace is collapsed, and tokens are produced by simple whitespace splitting. The vocabulary is built from the training split only with `min_freq=1`, `<pad>=0`, and `<unk>=1`. Sequences use `max_len=24` with post-truncation and post-padding. Labels are encoded alphabetically and saved with the vocabulary for reproducible inference.

| Hyperparameter | Value |
|---|---:|
| Seed | 42 |
| Split | 70/15/15 stratified |
| Max length | 24 |
| Embedding dimension | 128 |
| CNN kernels | 2, 3, 4 |
| CNN filters | 100 each |
| BiLSTM hidden size | 128 per direction |
| Embedding dropout | 0.3 |
| Classifier dropout | 0.4 |
| Loss | CrossEntropyLoss, label_smoothing=0.05 |
| Optimizer | AdamW |
| Learning rate | 0.002 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Max epochs | 60 |
| Early stopping | patience 8 on validation macro-F1 |
| Fallback threshold | 0.55 |

If the maximum softmax probability is below 0.55, the backend returns a graceful fallback response instead of pretending to understand. The fallback lists example supported questions.

## Model Architecture

The model is a hybrid CNN-BiLSTM intent classifier. Embeddings first pass through dropout. Three parallel Conv1d branches capture local n-gram evidence. In parallel, a bidirectional LSTM models token sequence context. Additive attention pools the BiLSTM output while masking padding positions. The pooled CNN features and attention-pooled recurrent context are concatenated, regularized, and classified.

| Layer | Output shape | Parameters |
|---|---|---:|
| Embedding | `(batch, 24, 128)` | 40,704 |
| Dropout | `(batch, 24, 128)` | 0 |
| Conv1d k=2 | `(batch, 100, 23)` | 25,700 |
| Conv1d k=3 | `(batch, 100, 22)` | 38,500 |
| Conv1d k=4 | `(batch, 100, 21)` | 51,300 |
| Max pooling + concat | `(batch, 300)` | 0 |
| BiLSTM | `(batch, 24, 256)` | 264,192 |
| Additive attention | `(batch, 256)` | 65,792 |
| Concat | `(batch, 556)` | 0 |
| Linear + ReLU | `(batch, 128)` | 71,296 |
| Linear output | `(batch, 22)` | 2,838 |

Total trainable parameters: 560,578 with a vocabulary size of 318.

The checkpoint stores the model config, so inference can rebuild the architecture without hardcoding dimensions. The forward method can also return attention weights; the API exposes the three highest-weighted non-padding tokens as "why" chips.

## Speech Recognition Module

CampusMate implements two speech-recognition paths. The primary path uses the browser Web Speech API with `lang="en-IN"`, `continuous=false`, and `interimResults=true`. It is fast, gives live interim transcripts, and avoids server compute, but browser support differs and the implementation may depend on the user's browser services. The fallback path records audio with `MediaRecorder`, uploads it to `/api/transcribe`, and uses faster-whisper `tiny.en` on CPU with int8 quantization. Whisper fallback is slower on first request because the model is lazy-loaded, but it supports browsers without Web Speech API and provides an offline server-side option after Docker build.

## Results

The training script writes final metrics to `artifacts/metrics.json`, training curves to `artifacts/training_curves.png`, and the evaluator writes `artifacts/confusion_matrix.png`. On the deterministic synthetic test split, the hybrid model achieved 100.00% accuracy, 1.000 macro-F1, and 1.000 weighted-F1. CPU inference latency was 2.23 ms per sample on the local verification machine. This very high score is plausible for the balanced synthetic dataset, but it should not be interpreted as real-world deployment accuracy.

| Model | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| CNN-only | 1.000 | 1.000 | Local n-gram ablation; 527,810 parameters |
| BiLSTM-only | 1.000 | 1.000 | Sequential ablation; 522,178 parameters |
| Hybrid | 1.000 | 1.000 | Deployed model; 560,578 parameters |

Per-class precision, recall, and F1 were all 1.00 on the test split. Supports were five samples for most classes, with six samples for `exam_schedule`, `hostel_rules`, and `scholarship_info`, because stratified splitting of 34 examples per class cannot divide perfectly into 15% test slices.

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| admission_eligibility | 1.00 | 1.00 | 1.00 | 5 |
| admission_process | 1.00 | 1.00 | 1.00 | 5 |
| attendance_policy | 1.00 | 1.00 | 1.00 | 5 |
| bot_identity | 1.00 | 1.00 | 1.00 | 5 |
| campus_wifi_support | 1.00 | 1.00 | 1.00 | 5 |
| course_registration | 1.00 | 1.00 | 1.00 | 5 |
| exam_results | 1.00 | 1.00 | 1.00 | 5 |
| exam_schedule | 1.00 | 1.00 | 1.00 | 6 |
| fee_payment | 1.00 | 1.00 | 1.00 | 5 |
| fee_structure | 1.00 | 1.00 | 1.00 | 5 |
| goodbye | 1.00 | 1.00 | 1.00 | 5 |
| greeting | 1.00 | 1.00 | 1.00 | 5 |
| grievance_redressal | 1.00 | 1.00 | 1.00 | 5 |
| hostel_allotment | 1.00 | 1.00 | 1.00 | 5 |
| hostel_rules | 1.00 | 1.00 | 1.00 | 6 |
| library_borrowing | 1.00 | 1.00 | 1.00 | 5 |
| library_timings | 1.00 | 1.00 | 1.00 | 5 |
| mess_menu | 1.00 | 1.00 | 1.00 | 5 |
| placement_info | 1.00 | 1.00 | 1.00 | 5 |
| scholarship_info | 1.00 | 1.00 | 1.00 | 6 |
| thanks | 1.00 | 1.00 | 1.00 | 5 |
| transport_bus | 1.00 | 1.00 | 1.00 | 5 |

## Discussion & Limitations

CampusMate is intentionally closed-domain. This makes it suitable for a campus helpdesk demonstration but not for open-ended university advising. Synthetic data enables a reproducible academic submission, yet it cannot fully represent real student language. Actual use would need anonymized campus query logs, more accents, code-switching, noisy audio, and more out-of-scope examples. The system has no multi-turn memory, so it cannot resolve follow-up questions like "what about tomorrow?" without repeated context. The response layer is template based, so it cannot access live university databases. Finally, speech recognition accuracy can vary by browser, microphone, background noise, and accent.

## Future Work

Future versions could add authenticated student-portal integrations, retrieval from official notices, multilingual ASR, a larger real-world dataset, explicit out-of-domain training examples, and multi-turn dialogue state. A confidence calibration step could improve fallback behavior. The attention-token display could be expanded into a more rigorous explainability view with saliency comparisons.

## How to Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/build_dataset.py
python training/train.py
python training/evaluate.py
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## References

- Vaswani et al., "Attention Is All You Need," 2017.
- Bahdanau, Cho, and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," 2015.
- PyTorch documentation.
- FastAPI documentation.
- faster-whisper project documentation.
- Web Speech API documentation.
