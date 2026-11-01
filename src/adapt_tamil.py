"""
src/adapt_tamil.py
==================
Phase 7 & 8: Tamil few-shot adaptation with four experimental variants.

Implements the PN-CLT research framework comparing:

  Model A — Tamil Baseline
    • Trained from scratch on Tamil data only
    • No transfer learning, no prosodic normalization
    • Establishes a lower-bound performance baseline

  Model B — Cross-Lingual Transfer (no normalization)
    • Load English pretrained weights
    • Freeze CNN backbone, fine-tune head + BiLSTM on Tamil
    • No prosodic normalization (raw prosodic features)

  Model C — Prosodic Normalization (no transfer)
    • Train from scratch on Tamil (like Model A)
    • BUT with Tamil language-adaptive prosodic normalization
    • Tests if normalization alone helps

  Model D — PN-CLT (PROPOSED METHOD)
    • Load English pretrained weights (transfer learning)
    • Apply Tamil language-adaptive prosodic normalization
    • Fine-tune using 2-stage strategy on Tamil data
    • Direct comparison with B shows benefit of normalization
    • Direct comparison with C shows benefit of transfer learning

Usage:
    python src/adapt_tamil.py

IMPORTANT WARNINGS:
    - With 1–10 Tamil files: demo/proof-of-concept only
    - Statistical metrics require adequate data (20+ per class minimum)
    - Results should NOT be reported as validated performance

Output:
    models/tamil_modelA.keras
    models/tamil_modelB.keras
    models/tamil_modelC.keras
    models/tamil_modelD.keras
    models/prosodic_normalizer_tamil.json
    results/figures/tamil_comparison.png
    results/reports/tamil_comparison.csv
"""

from __future__ import annotations

import os
import sys
import csv
import json
import warnings
import time
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

import tensorflow as tf
from tensorflow import keras

from src.config import cfg
from src.dataset import (
    build_tamil_dataset, detect_tamil_mode,
    load_label_encoder, scan_tamil, sample_few_shot,
)
from src.model import (
    build_model_from_config, compile_model, get_callbacks,
    freeze_layers, unfreeze_layers, print_trainability,
    replace_classification_head, FROZEN_CORE_LAYERS, UPPER_LAYERS,
)
from src.normalization import ProsodicNormalizer
from src.preprocessing import split_indices, load_audio
from src.feature_extraction import extract_features, normalize_spectrogram
from src.prosody import extract_prosodic_features
from src.normalization import append_prosody_to_spectrogram


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def adapt_tamil(cfg=cfg) -> dict:
    """
    Run Tamil adaptation for all four model variants.

    Returns
    -------
    dict: model_name -> {'accuracy', 'f1_macro', 'f1_weighted'}
    """
    cfg.ensure_dirs()

    print("\n" + "=" * 65)
    print("  PN-CLT: Tamil Few-Shot Adaptation")
    print("=" * 65)

    tamil_dir = Path(cfg.paths.tamil_data)
    mode, n_files = detect_tamil_mode(tamil_dir)

    # -------------------------------------------------------
    # Mode-specific dispatch
    # -------------------------------------------------------
    if mode == "demo" or n_files == 0:
        print("[adapt_tamil] Running DEMO MODE — inference only.")
        _run_demo_mode(cfg)
        return {}

    if mode == "poc":
        print("[adapt_tamil] Running PROOF-OF-CONCEPT MODE.")
        return _run_poc_mode(cfg, n_files)

    # few_shot or full
    return _run_full_adaptation(cfg, mode)


# ---------------------------------------------------------------------------
# DEMO MODE (1 file — inference only)
# ---------------------------------------------------------------------------

def _run_demo_mode(cfg) -> None:
    """Run inference on whatever Tamil files exist, using English model."""
    tamil_dir = Path(cfg.paths.tamil_data)
    wav_files = sorted(
        list(set(tamil_dir.rglob("*.wav")) | set(tamil_dir.rglob("*.WAV")))
    )

    if not wav_files:
        print("[demo] No Tamil WAV files found in", tamil_dir)
        print("       Place Tamil WAV files in data/tamil/ to run demo.")
        return

    print(f"\n[demo] Found {len(wav_files)} Tamil file(s). Running inference...")
    print("       (No fine-tuning — using English pretrained model)\n")

    # Load English model
    eng_path = Path(cfg.paths.english_best)
    if not eng_path.exists():
        print(f"[demo] English model not found: {eng_path}")
        print("       Run 'python src/train_english.py' first.")
        return

    model = keras.models.load_model(str(eng_path))
    le = load_label_encoder(cfg.paths.label_encoder)

    a, f, p = cfg.audio, cfg.features, cfg.prosody

    for wav_file in wav_files:
        print(f"  File: {wav_file.name}")
        try:
            audio = load_audio(wav_file, sr=a.sample_rate, duration=a.duration)
            spec = extract_features(audio, sr=a.sample_rate, n_mels=f.n_mels,
                                    n_fft=f.n_fft, hop_length=f.hop_length,
                                    fmin=f.fmin, fmax=f.fmax)
            spec = normalize_spectrogram(spec)
            pvec = extract_prosodic_features(audio, sr=a.sample_rate,
                                             f0_min=p.f0_min, f0_max=p.f0_max)
            combined = append_prosody_to_spectrogram(spec, pvec)
            X = combined[np.newaxis, ...]   # (1, T, F)
            probs = model.predict(X, verbose=0)[0]
            pred_idx = np.argmax(probs)
            pred_label = le.classes_[pred_idx]
            print(f"  Predicted: {pred_label.upper()} ({probs[pred_idx]*100:.1f}%)")
            for i, cls in enumerate(le.classes_):
                bar = "#" * int(probs[i] * 20)
                print(f"    {cls:<12} {bar:<20} {probs[i]*100:.1f}%")
            print()
        except Exception as exc:
            print(f"  ERROR: {exc}")


# ---------------------------------------------------------------------------
# PROOF-OF-CONCEPT MODE (2–19 files)
# ---------------------------------------------------------------------------

def _run_poc_mode(cfg, n_files: int) -> dict:
    """POC mode: run Model D inference on all files, no train/val split."""

    print(f"\n{'!'*65}")
    print("  PROOF-OF-CONCEPT MODE")
    print(f"  {n_files} Tamil files — NOT enough for reliable statistical evaluation.")
    print("  Results below are for demonstration purposes ONLY.")
    print(f"{'!'*65}\n")

    label_map = {k: v for k, v in vars(cfg.emotions.tamil_label_map).items()}

    # Build Tamil normalizer from all available files (statistically unreliable)
    X_all, y_all, file_names, mode = build_tamil_dataset(cfg=cfg, normalizer=None)

    if not X_all:
        print("[poc] No Tamil samples could be loaded.")
        return {}

    X_array = np.stack(X_all, axis=0)

    # Load English model for inference
    eng_path = Path(cfg.paths.english_best)
    if not eng_path.exists():
        print(f"[poc] English model not found. Run train_english.py first.")
        return {}

    model = keras.models.load_model(str(eng_path))
    le = load_label_encoder(cfg.paths.label_encoder)

    probs_all = model.predict(X_array, verbose=0)
    preds_all = np.argmax(probs_all, axis=1)

    print(f"\n  Per-file predictions (English model, no Tamil fine-tuning):")
    print(f"  {'File':<30} {'True':>10} {'Predicted':>12} {'Confidence':>12}")
    print("  " + "-" * 64)
    for i, fname in enumerate(file_names):
        true_lbl = le.classes_[y_all[i]] if y_all[i] >= 0 else "unknown"
        pred_lbl = le.classes_[preds_all[i]]
        conf = probs_all[i][preds_all[i]]
        print(f"  {fname[:30]:<30} {true_lbl:>10} {pred_lbl:>12} {conf*100:>11.1f}%")

    labelled_mask = np.array(y_all) >= 0
    if labelled_mask.sum() >= 2:
        y_true_labelled = np.array(y_all)[labelled_mask]
        y_pred_labelled = preds_all[labelled_mask]
        acc = accuracy_score(y_true_labelled, y_pred_labelled)
        f1 = f1_score(y_true_labelled, y_pred_labelled,
                      average="macro", zero_division=0)
        print(f"\n  POC Accuracy (labelled files): {acc:.2f}")
        print(f"  POC Macro F1 (labelled files): {f1:.2f}")
        print(f"\n  *** These values have NO statistical validity. ***")
        print(f"      They are based on {labelled_mask.sum()} samples.")
    else:
        print("\n  Too few labelled samples to compute metrics.")

    return {"poc_mode": True, "n_files": n_files}


# ---------------------------------------------------------------------------
# FULL ADAPTATION (few_shot or full mode)
# ---------------------------------------------------------------------------

def _run_full_adaptation(cfg, mode: str) -> dict:
    """Run all 4 model variants with proper train/val/test split."""

    results = {}

    # -------------------------------------------------------
    # Build Tamil dataset WITHOUT normalization first
    # (prosodic normalization applied per-model below)
    # -------------------------------------------------------
    label_map = {k: v for k, v in vars(cfg.emotions.tamil_label_map).items()}

    from src.preprocessing import load_audio
    from src.feature_extraction import extract_features, normalize_spectrogram
    from src.prosody import extract_prosodic_features
    from src.normalization import append_prosody_to_spectrogram

    tamil_dir = Path(cfg.paths.tamil_data)
    samples = scan_tamil(tamil_dir, label_map)
    labelled = [(p, e) for p, e in samples if e != "unknown"]

    if not labelled:
        print("[adapt] No labelled Tamil files found. Cannot run adaptation.")
        return {}

    le = load_label_encoder(cfg.paths.label_encoder)

    # Load all audio and extract raw features
    print("\n[adapt] Extracting Tamil features...")
    a, f, p = cfg.audio, cfg.features, cfg.prosody
    specs_all, pros_all, y_raw = [], [], []

    for fpath, emotion in labelled:
        try:
            audio = load_audio(fpath, sr=a.sample_rate, duration=a.duration)
            spec = extract_features(audio, sr=a.sample_rate, n_mels=f.n_mels,
                                    n_fft=f.n_fft, hop_length=f.hop_length,
                                    fmin=f.fmin, fmax=f.fmax)
            spec = normalize_spectrogram(spec)
            pvec = extract_prosodic_features(audio, sr=a.sample_rate,
                                             f0_min=p.f0_min, f0_max=p.f0_max,
                                             frame_length=p.frame_length,
                                             hop_length_s=p.hop_length_s)
            specs_all.append(spec)
            pros_all.append(pvec)
            label_idx = int(le.transform([emotion])[0]) if emotion in le.classes_ else -1
            y_raw.append(label_idx)
        except Exception as exc:
            warnings.warn(f"[adapt] Skipping {fpath.name}: {exc}")

    valid_mask = np.array(y_raw) >= 0
    specs_all  = [specs_all[i]  for i in range(len(specs_all))  if valid_mask[i]]
    pros_all   = [pros_all[i]   for i in range(len(pros_all))   if valid_mask[i]]
    y_all      = np.array([y_raw[i] for i in range(len(y_raw)) if valid_mask[i]])

    N = len(specs_all)
    if N < 6:
        print(f"[adapt] Only {N} labelled samples. Minimum 6 needed for split.")
        return {}

    # Split indices
    train_idx, val_idx, test_idx = split_indices(N, val_frac=0.2,
                                                  test_frac=0.2, seed=cfg.seed)

    # --- Fit Tamil ProsodicNormalizer on TRAIN prosodic ONLY ---
    pros_train_raw = np.stack([pros_all[i] for i in train_idx], axis=0)

    tamil_normalizer = ProsodicNormalizer(language="tamil", method="zscore")
    tamil_normalizer.fit(pros_train_raw)
    tamil_normalizer.save(cfg.paths.normalizer_tamil)
    print(tamil_normalizer.summary())

    # --- Build input tensors for each split ---
    def _build_X(idx_list, use_norm: bool):
        arrays = []
        for i in idx_list:
            pvec = pros_all[i]
            if use_norm:
                pvec = tamil_normalizer.transform(pvec.reshape(1, -1))[0]
            combined = append_prosody_to_spectrogram(specs_all[i], pvec)
            arrays.append(combined)
        return np.stack(arrays, axis=0)

    # Without normalization (Models A, B)
    X_train_raw = _build_X(train_idx, use_norm=False)
    X_val_raw   = _build_X(val_idx,   use_norm=False)
    X_test_raw  = _build_X(test_idx,  use_norm=False)

    # With normalization (Models C, D)
    X_train_norm = _build_X(train_idx, use_norm=True)
    X_val_norm   = _build_X(val_idx,   use_norm=True)
    X_test_norm  = _build_X(test_idx,  use_norm=True)

    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]
    y_test  = y_all[test_idx]

    input_shape = X_train_raw.shape[1:]
    n_classes = cfg.emotions.n_classes

    print(f"\n[adapt] Tamil split sizes — train:{len(train_idx)} "
          f"val:{len(val_idx)} test:{len(test_idx)}")

    # -------------------------------------------------------
    # MODEL A — Tamil Baseline (scratch, no norm)
    # -------------------------------------------------------
    print("\n" + "-" * 60)
    print("  MODEL A: Tamil Baseline (scratch, no normalization)")
    print("-" * 60)
    results["model_A"] = _train_from_scratch(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
        input_shape, n_classes, cfg, save_path=cfg.paths.model_A,
        model_tag="A",
    )

    # -------------------------------------------------------
    # MODEL B — Cross-Lingual Transfer (no norm)
    # -------------------------------------------------------
    print("\n" + "-" * 60)
    print("  MODEL B: Cross-Lingual Transfer (no normalization)")
    print("-" * 60)
    results["model_B"] = _fine_tune_model(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
        n_classes, cfg, save_path=cfg.paths.model_B, model_tag="B",
    )

    # -------------------------------------------------------
    # MODEL C — Prosodic Normalization (scratch, with norm)
    # -------------------------------------------------------
    print("\n" + "-" * 60)
    print("  MODEL C: Prosodic Normalization (scratch, with normalization)")
    print("-" * 60)
    results["model_C"] = _train_from_scratch(
        X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test,
        input_shape, n_classes, cfg, save_path=cfg.paths.model_C,
        model_tag="C",
    )

    # -------------------------------------------------------
    # MODEL D — PN-CLT: Transfer + Normalization (PROPOSED)
    # -------------------------------------------------------
    print("\n" + "-" * 60)
    print("  MODEL D: PN-CLT — Transfer + Normalization (PROPOSED)")
    print("-" * 60)
    results["model_D"] = _fine_tune_model(
        X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test,
        n_classes, cfg, save_path=cfg.paths.model_D, model_tag="D",
    )

    # -------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------
    _print_comparison_table(results)
    _save_comparison_csv(results, cfg.paths.reports_dir)
    _plot_comparison(results, cfg.paths.figures_dir)

    if mode == "full":
        _run_few_shot_curves(cfg, specs_all, pros_all, y_all,
                             tamil_normalizer, n_classes, le)

    return results


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_from_scratch(
    X_train, y_train, X_val, y_val, X_test, y_test,
    input_shape, n_classes, cfg, save_path, model_tag,
) -> dict:
    """Train a fresh CNN-BiLSTM on Tamil from random initialization."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model = build_model_from_config(cfg)
    compile_model(model, lr=cfg.fine_tuning.phase1_lr, n_classes=n_classes)

    callbacks = get_callbacks(
        checkpoint_path=save_path,
        patience_stop=cfg.fine_tuning.phase1_patience,
        patience_lr=5,
    )

    batch_size = min(cfg.fine_tuning.batch_size, max(1, len(X_train)))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.fine_tuning.phase1_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Load best (or fallback to model with restored weights)
    if Path(save_path).exists():
        best = keras.models.load_model(save_path)
    else:
        best = model
        model.save(save_path)
    return _evaluate(best, X_test, y_test, model_tag)


def _fine_tune_model(
    X_train, y_train, X_val, y_val, X_test, y_test,
    n_classes, cfg, save_path, model_tag,
) -> dict:
    """
    2-stage fine-tuning from English pretrained model.

    Stage 1: Freeze CNN core, train head only (fast adaptation)
    Stage 2: Unfreeze BiLSTM + head, fine-tune at very low LR
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    eng_path = Path(cfg.paths.english_best)
    if not eng_path.exists():
        print(f"  [{model_tag}] English model not found: {eng_path}")
        print(f"  [{model_tag}] Train English model first.")
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

    # Load pretrained English model
    print(f"  [{model_tag}] Loading English pretrained weights...")
    model = keras.models.load_model(str(eng_path))

    # Replace head if n_classes differs from English
    # (always replace to reinitialize — avoids English bias in head weights)
    model = replace_classification_head(model, n_classes=n_classes)

    # ---- Phase 1: Head only ----
    print(f"\n  [{model_tag}] Phase 1: Training head only (frozen CNN)...")
    freeze_layers(model, FROZEN_CORE_LAYERS)
    compile_model(model, lr=cfg.fine_tuning.phase1_lr, n_classes=n_classes)
    print_trainability(model)

    phase1_save = save_path.replace(".keras", "_phase1.keras")
    cbs1 = get_callbacks(
        checkpoint_path=phase1_save,
        patience_stop=cfg.fine_tuning.phase1_patience,
        patience_lr=5,
    )
    batch_size = min(cfg.fine_tuning.batch_size, max(1, len(X_train)))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.fine_tuning.phase1_epochs,
        batch_size=batch_size,
        callbacks=cbs1,
        verbose=1,
    )

    # Load phase 1 best
    if Path(phase1_save).exists():
        model = keras.models.load_model(phase1_save)
    else:
        model.save(phase1_save)

    # ---- Phase 2: Selective unfreeze ----
    print(f"\n  [{model_tag}] Phase 2: Fine-tuning with BiLSTM unfrozen...")
    unfreeze_layers(model, UPPER_LAYERS)
    compile_model(model, lr=cfg.fine_tuning.phase2_lr, n_classes=n_classes)
    print_trainability(model)

    cbs2 = get_callbacks(
        checkpoint_path=save_path,
        patience_stop=cfg.fine_tuning.phase2_patience,
        patience_lr=7,
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.fine_tuning.phase2_epochs,
        batch_size=batch_size,
        callbacks=cbs2,
        verbose=1,
    )

    # Load best phase 2
    if Path(save_path).exists():
        best = keras.models.load_model(save_path)
    else:
        best = model
        model.save(save_path)
    return _evaluate(best, X_test, y_test, model_tag)


def _evaluate(model, X_test, y_test, model_tag) -> dict:
    """Compute and print evaluation metrics."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    print(f"\n  [Model {model_tag}] Test: "
          f"Acc={acc:.4f} | Macro-F1={f1_macro:.4f} | Weighted-F1={f1_weighted:.4f}")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# ---------------------------------------------------------------------------
# Few-shot learning curves
# ---------------------------------------------------------------------------

def _run_few_shot_curves(cfg, specs_all, pros_all, y_all,
                          normalizer, n_classes, le) -> None:
    """
    Run few-shot experiments varying n_per_class.
    Compares Model B (no norm) vs Model D (PN-CLT) across shot counts.
    """
    print("\n" + "=" * 60)
    print("  FEW-SHOT LEARNING CURVES (Model B vs Model D)")
    print("=" * 60)
    print("  *** Results are illustrative, not statistically validated ***\n")

    shot_list = cfg.few_shot.samples_per_class_list
    n_repeats = cfg.few_shot.n_repeats

    # Build full tensors
    X_raw_all  = np.stack([append_prosody_to_spectrogram(specs_all[i], pros_all[i])
                           for i in range(len(specs_all))], axis=0)
    pros_array = np.stack(pros_all, axis=0)
    X_norm_all = np.stack([append_prosody_to_spectrogram(
                               specs_all[i],
                               normalizer.transform(pros_all[i].reshape(1,-1))[0])
                           for i in range(len(specs_all))], axis=0)

    results_B, results_D = [], []

    for n_shot in shot_list:
        if n_shot * n_classes > len(y_all) * 0.7:
            print(f"  Skipping n_shot={n_shot}: insufficient data.")
            continue

        f1_B_list, f1_D_list = [], []
        for rep in range(n_repeats):
            seed = cfg.seed + rep * 100
            # Sample few-shot training data
            X_few_raw,  y_few = sample_few_shot(X_raw_all,  y_all, n_shot, n_classes, seed)
            X_few_norm, _     = sample_few_shot(X_norm_all, y_all, n_shot, n_classes, seed)

            # Remaining as test
            selected_idx = _get_few_shot_indices(y_all, n_shot, n_classes, seed)
            test_mask = np.ones(len(y_all), dtype=bool)
            test_mask[selected_idx] = False
            if test_mask.sum() == 0:
                continue
            X_test_raw  = X_raw_all[test_mask]
            X_test_norm = X_norm_all[test_mask]
            y_test      = y_all[test_mask]

            # Quick fine-tune Model B (no norm)
            f1_B = _quick_fine_tune(X_few_raw,  y_few, X_test_raw,  y_test,
                                     n_classes, cfg, use_pretrained=True,
                                     use_norm=False)
            # Quick fine-tune Model D (norm)
            f1_D = _quick_fine_tune(X_few_norm, y_few, X_test_norm, y_test,
                                     n_classes, cfg, use_pretrained=True,
                                     use_norm=True)

            f1_B_list.append(f1_B)
            f1_D_list.append(f1_D)

        if f1_B_list:
            results_B.append((n_shot, np.mean(f1_B_list), np.std(f1_B_list)))
            results_D.append((n_shot, np.mean(f1_D_list), np.std(f1_D_list)))
            print(f"  n_shot={n_shot:3d}: "
                  f"B={np.mean(f1_B_list):.3f}+-{np.std(f1_B_list):.3f} | "
                  f"D={np.mean(f1_D_list):.3f}+-{np.std(f1_D_list):.3f}")

    if results_B:
        _plot_few_shot_curves(results_B, results_D, cfg.paths.figures_dir)


def _get_few_shot_indices(y, n_shot, n_classes, seed):
    """Get indices of few-shot selected samples."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        n_take = min(n_shot, len(cls_idx))
        if n_take > 0:
            idx.extend(rng.choice(cls_idx, size=n_take, replace=False).tolist())
    return idx


def _quick_fine_tune(X_train, y_train, X_test, y_test,
                      n_classes, cfg, use_pretrained, use_norm) -> float:
    """Quick fine-tune for few-shot curve (head-only, minimal epochs)."""
    if use_pretrained and Path(cfg.paths.english_best).exists():
        model = keras.models.load_model(str(cfg.paths.english_best))
        model = replace_classification_head(model, n_classes=n_classes)
        freeze_layers(model, FROZEN_CORE_LAYERS)
    else:
        model = build_model_from_config(cfg)

    compile_model(model, lr=cfg.fine_tuning.phase1_lr, n_classes=n_classes)

    if len(X_train) == 0:
        return 0.0

    model.fit(X_train, y_train, epochs=20, batch_size=max(1, len(X_train)//4),
              verbose=0, validation_split=0.0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return float(f1_score(y_test, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Plotting and reporting
# ---------------------------------------------------------------------------

def _print_comparison_table(results: dict) -> None:
    print("\n" + "=" * 65)
    print("  TAMIL ADAPTATION — COMPARISON TABLE")
    print("  *** Proof-of-concept: interpret with caution ***")
    print("=" * 65)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print("  " + "-" * 52)
    labels = {
        "model_A": "A: Baseline",
        "model_B": "B: Transfer",
        "model_C": "C: Norm Only",
        "model_D": "D: PN-CLT (Proposed)",
    }
    for key, label in labels.items():
        if key in results:
            r = results[key]
            print(f"  {label:<20} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
                  f"{r['f1_weighted']:>12.4f}")
    print("=" * 65)


def _save_comparison_csv(results: dict, reports_dir: str) -> None:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    path = Path(reports_dir) / "tamil_comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "f1_macro", "f1_weighted"])
        writer.writeheader()
        for key, r in results.items():
            if isinstance(r, dict) and "accuracy" in r:
                writer.writerow({"model": key, **{k: f"{v:.4f}" for k, v in r.items()}})
    print(f"\n[adapt] Comparison CSV saved: {path}")


def _plot_comparison(results: dict, figures_dir: str) -> None:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    labels, accs, f1s = [], [], []
    label_map = {"model_A": "A\nBaseline", "model_B": "B\nTransfer",
                 "model_C": "C\nNorm", "model_D": "D\nPN-CLT"}
    for key, lab in label_map.items():
        if key in results and "accuracy" in results[key]:
            labels.append(lab)
            accs.append(results[key]["accuracy"])
            f1s.append(results[key]["f1_macro"])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tamil Adaptation — Model Comparison\n"
                 "(⚠ Proof-of-concept: limited data)",
                 fontsize=12, fontweight="bold")

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    axes[0].bar(x, accs, width, color=colors[:len(labels)], alpha=0.8)
    axes[0].set_title("Accuracy")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    axes[1].bar(x, f1s, width, color=colors[:len(labels)], alpha=0.8)
    axes[1].set_title("Macro F1")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    out = Path(figures_dir) / "tamil_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[adapt] Comparison plot saved: {out}")


def _plot_few_shot_curves(results_B, results_D, figures_dir: str) -> None:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    shots_B = [r[0] for r in results_B]
    f1_B    = [r[1] for r in results_B]
    err_B   = [r[2] for r in results_B]
    shots_D = [r[0] for r in results_D]
    f1_D    = [r[1] for r in results_D]
    err_D   = [r[2] for r in results_D]

    plt.figure(figsize=(9, 5))
    plt.errorbar(shots_B, f1_B, yerr=err_B, fmt="b-o", label="Model B (Transfer, no norm)",
                 capsize=4, linewidth=2, markersize=6)
    plt.errorbar(shots_D, f1_D, yerr=err_D, fmt="r-s", label="Model D (PN-CLT, proposed)",
                 capsize=4, linewidth=2, markersize=6)
    plt.title("Few-Shot Learning Curves: B vs D\n(error bars = std over repeats)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Samples per Class")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(figures_dir) / "few_shot_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[adapt] Few-shot curve saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = adapt_tamil()
    if results and not results.get("poc_mode"):
        print("\nAdaptation complete. Run 'python src/evaluate.py' for full evaluation.")
