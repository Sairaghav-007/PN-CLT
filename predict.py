"""
predict.py
==========
CLI tool to predict emotion from a single audio file using a trained PN-CLT model.

Usage:
    python predict.py --audio path/to/sample.wav
    python predict.py --audio path/to/sample.wav --model models/tamil_modelD.keras --lang tamil
    python predict.py --audio path/to/sample.wav --model models/english_best.keras --lang english
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.preprocessing import load_audio, pad_or_truncate
from src.feature_extraction import extract_features, normalize_spectrogram
from src.prosody import extract_prosodic_features
from src.normalization import ProsodicNormalizer, append_prosody_to_spectrogram
from src.dataset import load_label_encoder

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("[ERROR] TensorFlow is required for inference.")
    sys.exit(1)


def predict_audio(
    audio_path: str | Path,
    model_path: str | Path = "models/tamil_modelD.keras",
    normalizer_path: str | Path | None = None,
    language: str = "tamil",
    label_encoder_path: str | Path = "models/label_encoder.pkl",
    top_k: int = 3,
):
    audio_path = Path(audio_path)
    model_path = Path(model_path)
    label_encoder_path = Path(label_encoder_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run training or adaptation first.")

    print(f"\n[1/4] Loading audio: {audio_path.name}")
    audio = load_audio(audio_path, sr=cfg.audio.sample_rate, duration=cfg.audio.duration)

    print(f"[2/4] Extracting Log-Mel and Prosodic features...")
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
    pros = extract_prosodic_features(
        audio,
        sr=cfg.audio.sample_rate,
        f0_min=cfg.prosody.f0_min,
        f0_max=cfg.prosody.f0_max,
        frame_length=cfg.prosody.frame_length,
        hop_length_s=cfg.prosody.hop_length_s,
    )

    # Prosodic normalization
    if normalizer_path is None:
        if language.lower() == "tamil":
            normalizer_path = Path("models/prosodic_normalizer_tamil.pkl")
        else:
            normalizer_path = Path("models/prosodic_normalizer_english.pkl")

    if Path(normalizer_path).exists():
        print(f"[3/4] Applying prosodic normalizer: {normalizer_path.name}")
        normalizer = ProsodicNormalizer.load(normalizer_path)
        pros = normalizer.transform(pros.reshape(1, -1))[0]
    else:
        print(f"[3/4] Normalizer {normalizer_path} not found; proceeding without normalization.")

    # Combine
    X = append_prosody_to_spectrogram(spec, pros)
    X = np.expand_dims(X, axis=0)  # (1, TIME_STEPS, FEATURES)

    print(f"[4/4] Running inference with: {model_path.name}")
    model = keras.models.load_model(str(model_path), compile=False)
    probs = model.predict(X, verbose=0)[0]

    # Load label encoder
    if label_encoder_path.exists():
        le = load_label_encoder(label_encoder_path)
        classes = list(le.classes_)
    else:
        classes = [f"Class_{i}" for i in range(len(probs))]

    # Sort probabilities
    top_indices = np.argsort(probs)[::-1][:top_k]

    print("\n" + "=" * 45)
    print(f"  AUDIO:      {audio_path.name}")
    print(f"  PREDICTION: {classes[top_indices[0]].upper()} ({probs[top_indices[0]] * 100:.1f}%)")
    print("=" * 45)
    print("  Top Predicted Emotions:")
    for rank, idx in enumerate(top_indices, start=1):
        bar = "#" * int(probs[idx] * 25)
        print(f"   {rank}. {classes[idx]:<10} {probs[idx] * 100:>5.1f}%  [{bar}]")
    print("=" * 45 + "\n")

    return classes[top_indices[0]], probs[top_indices[0]]


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file using PN-CLT.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--model", type=str, default="models/tamil_modelD.keras", help="Path to trained model (.keras)")
    parser.add_argument("--lang", type=str, default="tamil", choices=["tamil", "english"], help="Language normalizer to use")
    parser.add_argument("--normalizer", type=str, default=None, help="Custom path to normalizer file")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")

    args = parser.parse_args()
    predict_audio(
        audio_path=args.audio,
        model_path=args.model,
        normalizer_path=args.normalizer,
        language=args.lang,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
