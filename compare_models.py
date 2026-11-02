"""
compare_models.py
=================
Demonstration tool for panel presentation:
Runs a single audio file through all 4 models (Model A, B, C, and D)
and displays a side-by-side comparison table showing how PN-CLT (Model D)
improves emotion classification compared to the ablation baselines.

Usage:
    python compare_models.py --audio data/tamil/angry/audio_12.wav
    python compare_models.py --audio path/to/your_recorded_tamil_voice.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Suppress TF logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("[ERROR] TensorFlow is required.")
    sys.exit(1)

from src.config import cfg
from src.preprocessing import load_audio
from src.feature_extraction import extract_features, normalize_spectrogram
from src.prosody import extract_prosodic_features
from src.normalization import ProsodicNormalizer, append_prosody_to_spectrogram
from src.dataset import load_label_encoder


def compare_all_models(audio_path: str | Path):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return

    print("=" * 80)
    print("      PN-CLT SER: 4-MODEL LIVE DEMONSTRATION & ABLATION COMPARISON")
    print("=" * 80)
    print(f"  Target Audio File: {audio_path.name}")
    print(f"  Path:             {audio_path.resolve()}\n")

    # Step 1: Preprocess Audio & Extract Features
    print("[1/3] Preprocessing audio and extracting Log-Mel & Prosodic features...")
    audio = load_audio(audio_path, sr=cfg.audio.sample_rate, duration=cfg.audio.duration)
    spec = extract_features(
        audio,
        sr=cfg.audio.sample_rate,
        n_mels=cfg.features.n_mels,
        n_fft=cfg.features.n_fft,
        hop_length=cfg.features.hop_length,
        fmin=cfg.features.fmin,
        fmax=cfg.features.fmax,
        use_log=getattr(cfg.features, "use_log_mel", True),
    )
    spec = normalize_spectrogram(spec)
    pros_raw = extract_prosodic_features(
        audio,
        sr=cfg.audio.sample_rate,
        f0_min=cfg.prosody.f0_min,
        f0_max=cfg.prosody.f0_max,
        frame_length=cfg.prosody.frame_length,
        hop_length_s=cfg.prosody.hop_length_s,
    )

    # Step 2: Build Unnormalized and Normalized Input Tensors
    print("[2/3] Preparing unnormalized (A, B) and normalized (C, D) input representations...")
    # Unnormalized tensor for Models A & B
    X_raw = append_prosody_to_spectrogram(spec, pros_raw)
    X_raw = np.expand_dims(X_raw, axis=0)

    # Normalized tensor for Models C & D
    norm_path = Path("models/prosodic_normalizer_tamil.pkl")
    if norm_path.exists():
        normalizer = ProsodicNormalizer.load(norm_path)
        pros_norm = normalizer.transform(pros_raw.reshape(1, -1))[0]
    else:
        print("[WARNING] Tamil normalizer not found; using raw features.")
        pros_norm = pros_raw

    X_norm = append_prosody_to_spectrogram(spec, pros_norm)
    X_norm = np.expand_dims(X_norm, axis=0)

    # Load label encoder
    le_path = Path("models/label_encoder.pkl")
    if le_path.exists():
        le = load_label_encoder(le_path)
        classes = list(le.classes_)
    else:
        classes = [f"Class_{i}" for i in range(7)]

    # Model definitions
    model_configs = [
        {
            "tag": "Model A",
            "name": "Tamil Baseline",
            "transfer": "No (Scratch)",
            "norm": "No",
            "path": Path("models/tamil_modelA.keras"),
            "input": X_raw,
            "desc": "Baseline trained on Tamil from scratch without normalization",
        },
        {
            "tag": "Model B",
            "name": "Cross-Lingual Transfer",
            "transfer": "Yes (English)",
            "norm": "No",
            "path": Path("models/tamil_modelB.keras"),
            "input": X_raw,
            "desc": "Transferred from English without prosodic normalization",
        },
        {
            "tag": "Model C",
            "name": "Prosodic Norm Alone",
            "transfer": "No (Scratch)",
            "norm": "Yes (Tamil)",
            "path": Path("models/tamil_modelC.keras"),
            "input": X_norm,
            "desc": "Trained from scratch on Tamil with prosodic normalization",
        },
        {
            "tag": "Model D",
            "name": "PN-CLT (Proposed)",
            "transfer": "Yes (English)",
            "norm": "Yes (Tamil)",
            "path": Path("models/tamil_modelD.keras"),
            "input": X_norm,
            "desc": "English Pretrained + Tamil Prosodic Normalization + 2-Stage Transfer",
        },
    ]

    print("[3/3] Evaluating across all 4 models in memory...\n")
    results = []

    for cfg_m in model_configs:
        m_path = cfg_m["path"]
        if not m_path.exists():
            results.append({
                "tag": cfg_m["tag"],
                "name": cfg_m["name"],
                "transfer": cfg_m["transfer"],
                "norm": cfg_m["norm"],
                "prediction": "MODEL NOT FOUND",
                "confidence": 0.0,
                "all_probs": {},
            })
            continue

        model = keras.models.load_model(str(m_path), compile=False)
        probs = model.predict(cfg_m["input"], verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_class = classes[top_idx]
        conf = float(probs[top_idx]) * 100.0

        all_probs = {classes[i]: float(probs[i]) * 100.0 for i in range(len(classes))}
        results.append({
            "tag": cfg_m["tag"],
            "name": cfg_m["name"],
            "transfer": cfg_m["transfer"],
            "norm": cfg_m["norm"],
            "prediction": top_class.upper(),
            "confidence": conf,
            "all_probs": all_probs,
        })

    # Display Side-by-Side Summary Table
    print("=" * 86)
    print(f"{'MODEL':<10} {'METHOD / ARCHITECTURE':<26} {'TRANSFER?':<14} {'NORM?':<12} {'PREDICTION':<12} {'CONFIDENCE':>10}")
    print("-" * 86)
    for r in results:
        marker = " <== PROPOSED" if r["tag"] == "Model D" else ""
        print(f"{r['tag']:<10} {r['name']:<26} {r['transfer']:<14} {r['norm']:<12} {r['prediction']:<12} {r['confidence']:>9.1f}%{marker}")
    print("=" * 86)

    # Detailed Class Probabilities Breakdown
    print("\n" + "-" * 86)
    print("                       FULL EMOTION PROBABILITY BREAKDOWN")
    print("-" * 86)
    header = f"{'Emotion':<12}" + "".join([f"{r['tag']:>18}" for r in results])
    print(header)
    print("-" * 86)
    for cls in classes:
        row = f"{cls:<12}"
        for r in results:
            val = r["all_probs"].get(cls, 0.0)
            row += f"{val:>17.1f}%"
        print(row)
    print("-" * 86)

    print("\n[*] PANEL EXPLANATION KEY POINTS:")
    print(" - Model A: Trained from scratch on limited Tamil data; lacks representation power.")
    print(" - Model B: Transfers English weights without normalization; suffers from cross-lingual prosody mismatch.")
    print(" - Model C: Normalizes prosody but trained from scratch; lacks pretrained acoustic features.")
    print(" - Model D (PN-CLT): Combines English transfer learning with Tamil prosodic normalization and 2-stage fine-tuning.")
    print("=" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run side-by-side comparison of Models A, B, C, D on an audio file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav, .mp3, etc.)")
    args = parser.parse_args()
    compare_all_models(args.audio)


if __name__ == "__main__":
    main()
