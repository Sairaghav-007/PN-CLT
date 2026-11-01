"""
dry_run_tamil.py
================
Dry-run script for PN-CLT Speech Emotion Recognition (SER) pipeline.
Selects exactly 1 audio sample from each Tamil emotion class and passes them
through the complete end-to-end pipeline:
  1. Audio Loading & Preprocessing (16kHz, mono, 3.0s pad/truncation)
  2. Log-Mel Spectrogram Extraction (94 time steps x 128 mel bins)
  3. Prosodic Feature Extraction (12 prosodic features)
  4. Tamil Language-Adaptive Prosodic Normalization (z-score)
  5. Feature Fusion (Spectrogram + Normalized Prosody -> 94 x 140)
  6. Model Inference (PN-CLT Model D: models/tamil_modelD.keras)
  7. Performance & Prediction Evaluation
"""

import os
import sys
import time
from pathlib import Path

# Suppress TF logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

# Ensure project root in sys.path
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
    print("[ERROR] TensorFlow is required.")
    sys.exit(1)


def run_dry_run():
    print("=" * 70)
    print("  PN-CLT SER: TAMIL EMOTIONS DRY RUN (1 SAMPLE PER EMOTION)")
    print("=" * 70)

    tamil_data_dir = PROJECT_ROOT / "data" / "tamil"
    if not tamil_data_dir.exists():
        print(f"[ERROR] Tamil data directory not found: {tamil_data_dir}")
        return

    # Find emotion subdirectories
    emotion_dirs = sorted([d for d in tamil_data_dir.iterdir() if d.is_dir()])
    if not emotion_dirs:
        print(f"[ERROR] No emotion folders found in {tamil_data_dir}")
        return

    # Select exactly 1 sample from each emotion
    selected_samples = []
    for edir in emotion_dirs:
        wav_files = sorted(list(edir.glob("*.wav")) + list(edir.glob("*.WAV")))
        if wav_files:
            selected_samples.append((edir.name, wav_files[0]))
        else:
            print(f"[WARN] No wav files found in {edir.name}")

    print(f"\n[INFO] Found {len(selected_samples)} Tamil emotion classes:")
    for emotion, path in selected_samples:
        print(f"  - {emotion:<10}: {path.name} ({path.relative_to(PROJECT_ROOT)})")

    # Load resources
    model_path = PROJECT_ROOT / "models" / "tamil_modelD.keras"
    normalizer_path = PROJECT_ROOT / "models" / "prosodic_normalizer_tamil.pkl"
    label_encoder_path = PROJECT_ROOT / "models" / "label_encoder.pkl"

    print("\n[INFO] Loading Model & Pipeline Artifacts...")
    print(f"  * Model:            {model_path.name}")
    print(f"  * Normalizer:       {normalizer_path.name}")
    print(f"  * Label Encoder:    {label_encoder_path.name}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer not found: {normalizer_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

    model = keras.models.load_model(str(model_path), compile=False)
    normalizer = ProsodicNormalizer.load(normalizer_path)
    label_encoder = load_label_encoder(label_encoder_path)
    classes = list(label_encoder.classes_)

    print(f"  * Output classes ({len(classes)}): {classes}")
    print("\n" + "-" * 70)
    print(f"{'#':<3} {'Emotion (GT)':<14} {'File':<14} {'Audio Shape':<12} {'Input Shape':<14} {'Status'}")
    print("-" * 70)

    results = []
    total_start = time.time()

    for idx, (ground_truth, file_path) in enumerate(selected_samples, start=1):
        step_start = time.time()

        # Step 1: Preprocessing
        audio = load_audio(file_path, sr=cfg.audio.sample_rate, duration=cfg.audio.duration)
        audio_shape_str = f"{len(audio)} smp"

        # Step 2: Mel Spectrogram
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

        # Step 3: Prosody Extraction & Normalization
        pros = extract_prosodic_features(
            audio,
            sr=cfg.audio.sample_rate,
            f0_min=cfg.prosody.f0_min,
            f0_max=cfg.prosody.f0_max,
            frame_length=cfg.prosody.frame_length,
            hop_length_s=cfg.prosody.hop_length_s,
        )
        pros_norm = normalizer.transform(pros.reshape(1, -1))[0]

        # Step 4: Feature Fusion
        X = append_prosody_to_spectrogram(spec, pros_norm)
        input_shape_str = f"{X.shape[0]}x{X.shape[1]}"
        X_batch = np.expand_dims(X, axis=0)

        # Step 5: Model Inference
        probs = model.predict(X_batch, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_emotion = classes[pred_idx]
        pred_conf = probs[pred_idx] * 100.0

        elapsed_ms = (time.time() - step_start) * 1000

        top3_indices = np.argsort(probs)[::-1][:3]
        top3_str = ", ".join([f"{classes[i]}: {probs[i]*100:.1f}%" for i in top3_indices])

        is_match = (ground_truth.lower() == pred_emotion.lower())
        match_str = "MATCH" if is_match else "DIFF"

        results.append({
            "idx": idx,
            "gt": ground_truth,
            "file": file_path.name,
            "audio_len": len(audio),
            "input_shape": X.shape,
            "pred": pred_emotion,
            "conf": pred_conf,
            "match": is_match,
            "top3": top3_str,
            "probs": probs,
            "latency_ms": elapsed_ms,
            "prosody_sample": pros_norm[:4],  # Sample first 4 normalized features
        })

        print(f"{idx:<3} {ground_truth:<14} {file_path.name:<14} {audio_shape_str:<12} {input_shape_str:<14} [OK] {elapsed_ms:.1f}ms")

    total_time = time.time() - total_start

    print("-" * 70)
    print("\n" + "=" * 70)
    print("                      DETAILED INFERENCE RESULTS")
    print("=" * 70)

    for r in results:
        status_symbol = "[MATCH]" if r["match"] else "[NOTE]"
        print(f"\nSample {r['idx']}: Ground Truth: {r['gt'].upper()} | File: {r['file']}")
        print(f"  * Pipeline latency:      {r['latency_ms']:.2f} ms")
        print(f"  * Final input shape:     (1, {r['input_shape'][0]}, {r['input_shape'][1]})")
        print(f"  * Normalized prosody:    [F0_mean={r['prosody_sample'][0]:.2f}, F0_std={r['prosody_sample'][1]:.2f}, F0_range={r['prosody_sample'][2]:.2f}, voiced_frac={r['prosody_sample'][3]:.2f}]")
        print(f"  * Model Prediction:      {status_symbol} {r['pred'].upper()} ({r['conf']:.2f}%)")
        print(f"  * Top-3 Probabilities:   {r['top3']}")

    # Summary table
    print("\n" + "=" * 70)
    print("                           SUMMARY TABLE")
    print("=" * 70)
    print(f"{'#':<3} {'True Emotion':<14} {'Sample File':<14} {'Predicted':<12} {'Confidence':<12} {'Top-3 Predictions'}")
    print("-" * 70)
    for r in results:
        print(f"{r['idx']:<3} {r['gt']:<14} {r['file']:<14} {r['pred']:<12} {r['conf']:>5.1f}%      {r['top3']}")
    print("-" * 70)

    matches = sum(1 for r in results if r["match"])
    print(f"\n[COMPLETION SUMMARY]")
    print(f"  * Total Tamil emotions tested:      {len(results)}")
    print(f"  * Pipeline completion:              100% ({len(results)}/{len(results)} samples processed successfully)")
    print(f"  * Correct predictions:              {matches}/{len(results)} ({matches/len(results)*100:.1f}%)")
    print(f"  * Average latency per sample:       {sum(r['latency_ms'] for r in results)/len(results):.1f} ms")
    print(f"  * Total dry run execution time:     {total_time:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    run_dry_run()
