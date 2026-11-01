# Prosody-Normalized Cross-Lingual Transfer (PN-CLT)

A lightweight few-shot framework for Cross-Lingual Speech Emotion Recognition (SER) in low-resource Indian regional languages.

---

## 📌 Overview

Speech Emotion Recognition (SER) models trained on Western/English speech corpora (such as TESS or SAVEE) experience severe performance degradation when directly evaluated on Indian regional languages due to substantial differences in pitch, intonation contours, speaking rate, and prosody.

**PN-CLT** addresses this fundamental gap by combining:
1. **Language-Adaptive Prosodic Normalization**: Aligning target-language emotional prosody against native speaking baselines (Z-Score / Robust IQR).
2. **Hybrid CNN-BiLSTM Feature Fusion**: Jointly processing 128-dimensional Log-Mel spectrograms (MFSC) and 12-dimensional normalized prosodic context vectors ($94 \times 140$).
3. **2-Stage Few-Shot Cross-Lingual Transfer Learning**: Fine-tuning frozen acoustic representations from English to low-resource languages (e.g., Tamil) with minimal calibration samples.

---

## 🔬 Experimental Models

The framework implements four comparative models to isolate the precise contribution of each component:

- **Model A (Tamil Baseline)**: Trained from scratch on Tamil data only; no transfer learning and no prosodic normalization.
- **Model B (Cross-Lingual Transfer)**: Loads English pretrained weights and fine-tunes top layers without prosodic normalization.
- **Model C (Prosodic Normalization Alone)**: Trained from scratch on Tamil with language-adaptive prosodic normalization, isolating the impact of normalization without transfer.
- **Model D (PN-CLT — Proposed Method)**: Combines English pretraining + language-adaptive prosodic normalization + **2-Stage Fine-Tuning** (Stage 1: frozen CNN core, adapt classification head; Stage 2: selective unfreeze of BiLSTM at low learning rate).

---

## 🏗️ Architecture

```
Input Audio (16kHz, mono, 3.0s)
   │
   ├─► Log-Mel Spectrogram (MFSC) ──────► (94, 128)
   │                                           │
   └─► 12-D Prosodic Features                  ▼
       (Pitch, RMS, ZCR, Slopes)          Concatenation / Fusion
            │                                  │
       Prosodic Normalizer                     ▼
       z = (x - μ_lang) / (σ_lang + ε)    Combined Tensor (94, 140)
            │                                  │
            └──────────────────────────────────┘
                                               │
                                               ▼
                                      Block 1: Conv1D(128, k=5) + BN + ReLU + MaxPool + Dropout
                                               │
                                               ▼
                                      Block 2: Conv1D(256, k=3) + BN + ReLU + MaxPool + Dropout
                                               │
                                               ▼
                                      Temporal: Bidirectional LSTM(128)
                                               │
                                               ▼
                                      Dense(128) -> Dropout(0.4) -> Dense(64) -> Dropout(0.3)
                                               │
                                               ▼
                                      Output: Dense(n_classes, Softmax)
```

---

## 📁 Repository Structure

```
├── config.yaml                    # Master project configuration & hyperparameters
├── requirements.txt               # Python package dependencies
├── setup.cfg                      # Flake8 and pytest configuration
├── PN-CLT_Research_Document.md    # Methodology & research document
├── predict.py                     # CLI inference tool for single audio files
├── dry_run_tamil.py               # End-to-end multi-emotion verification pipeline
├── phase2_smoke_test.py           # Environment sanity check script
├── scripts/
│   └── download_sample_data.py    # Automated dataset downloader (TESS / Thanushs25)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Dataclass configuration loader
│   ├── preprocessing.py           # Resampling, mono conversion, silence detection, padding
│   ├── feature_extraction.py      # Log-Mel (MFSC) spectral feature extraction
│   ├── prosody.py                 # 12-D prosodic feature vector extraction
│   ├── normalization.py           # Language-adaptive prosodic normalizer & fusion
│   ├── model.py                   # CNN-BiLSTM architecture & transfer layer groups
│   ├── dataset.py                 # Dataset loaders & split management
│   ├── train_english.py           # English base model pretraining
│   └── adapt_tamil.py             # 4-model adaptation orchestrator (Models A, B, C, D)
└── tests/
    ├── __init__.py
    ├── conftest.py                # Synthetic audio test fixtures
    └── test_core_pipeline.py      # Automated unit tests
```

---

## 🚀 Quick Start

### 1. Installation
Clone repository and install dependencies:
```bash
git clone https://github.com/Sairaghav-007/PN-CLT.git
cd PN-CLT
pip install -r requirements.txt
```

Verify your environment:
```bash
python phase2_smoke_test.py
```

### 2. Download Datasets
Download representative samples of English (TESS) and Tamil speech emotion datasets:
```bash
python scripts/download_sample_data.py
```

### 3. Pretrain English Source Model
Train the base CNN-BiLSTM model on English:
```bash
python src/train_english.py
```

### 4. Run Few-Shot Adaptation (Models A, B, C, D)
Fine-tune and compare all four model variants:
```bash
python src/adapt_tamil.py
```

### 5. Single File Inference
Predict emotions on an audio file:
```bash
python predict.py --audio path/to/sample.wav --model models/tamil_modelD.keras --lang tamil
```

### 6. Run Unit Tests
```bash
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the Apache 2.0 / MIT License.
