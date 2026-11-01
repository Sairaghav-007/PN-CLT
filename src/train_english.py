"""
src/train_english.py
====================
Phase 6: English CNN-BiLSTM pretraining.

Usage:
    python src/train_english.py

What this script does:
  1. Loads config from config.yaml
  2. Builds English dataset from TESS (+ optional SAVEE)
  3. Builds CNN-BiLSTM model
  4. Trains with early stopping, LR scheduling, checkpointing
  5. Evaluates on held-out test set
  6. Saves: best model, final model, label encoder, training curves

Dataset required:
    Download TESS from:
    https://tspace.library.utoronto.ca/handle/1807/24487

    Extract to: data/english/TESS/
    Expected structure:
        data/english/TESS/OAF_angry/OAF_angry_01.wav
        data/english/TESS/YAF_happy/YAF_happy_01.wav
        ...

Output:
    models/english_best.keras    (best checkpoint)
    models/english_final.keras   (after all epochs)
    models/label_encoder.pkl
    models/preprocessing_config.json
    models/prosodic_normalizer_english.json
    results/figures/english_training_curves.png
    results/figures/english_confusion_matrix.png
    results/reports/english_classification_report.txt
"""

from __future__ import annotations

import os
import sys
import json
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to path (so imports work from any directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

import tensorflow as tf
from tensorflow import keras

from src.config import cfg
from src.dataset import build_english_dataset, load_label_encoder
from src.model import (
    build_model_from_config, compile_model, get_callbacks, print_trainability
)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_english(cfg=cfg) -> dict:
    """
    Train the English CNN-BiLSTM model.

    Returns
    -------
    dict with keys: 'test_accuracy', 'test_f1_macro', 'history'
    """
    cfg.ensure_dirs()
    start_time = time.time()

    print("\n" + "=" * 65)
    print("  PN-CLT: English CNN-BiLSTM Pretraining")
    print("=" * 65)
    print(f"  Seed          : {cfg.seed}")
    print(f"  Input shape   : {cfg.get_input_shape()}")
    print(f"  Classes       : {cfg.emotions.n_classes}")
    print(f"  TESS dir      : {cfg.paths.english_tess}")
    print("=" * 65 + "\n")

    # ---------------------------------------------------------------
    # Step 1: Build dataset
    # ---------------------------------------------------------------
    print("[Step 1/5] Building English dataset...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test, le) = build_english_dataset(cfg=cfg, verbose=True)

    print(f"\n  Class distribution (train):")
    for cls_idx, cls_name in enumerate(le.classes_):
        count = int(np.sum(y_train == cls_idx))
        print(f"    {cls_name:<12}: {count}")

    # ---------------------------------------------------------------
    # Step 2: Build model
    # ---------------------------------------------------------------
    print("\n[Step 2/5] Building CNN-BiLSTM model...")
    model = build_model_from_config(cfg)
    compile_model(model, lr=cfg.training.learning_rate,
                  n_classes=cfg.emotions.n_classes)
    model.summary(line_length=70)
    print_trainability(model)

    # ---------------------------------------------------------------
    # Step 3: Train
    # ---------------------------------------------------------------
    print("\n[Step 3/5] Training...")
    callbacks = get_callbacks(
        checkpoint_path=cfg.paths.english_best,
        patience_stop=cfg.training.early_stopping_patience,
        patience_lr=cfg.training.lr_reduce_patience,
        lr_factor=cfg.training.lr_reduce_factor,
        min_lr=cfg.training.min_lr,
    )

    batch_size = min(cfg.training.batch_size, max(1, len(X_train)))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.training.epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------------------
    # Step 4: Save final model
    # ---------------------------------------------------------------
    print(f"\n[Step 4/5] Saving models...")
    Path(cfg.paths.english_final).parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.paths.english_final)
    if not Path(cfg.paths.english_best).exists():
        model.save(cfg.paths.english_best)
    print(f"  Final model  -> {cfg.paths.english_final}")
    print(f"  Best model   -> {cfg.paths.english_best}")

    # Save training history
    history_path = Path(cfg.paths.results_dir) / "reports" / "english_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"  Training history -> {history_path}")

    # ---------------------------------------------------------------
    # Step 5: Evaluate on test set
    # ---------------------------------------------------------------
    print("\n[Step 5/5] Evaluating on test set...")

    # Load best checkpoint for evaluation (not the final weights)
    best_path = Path(cfg.paths.english_best)
    best_model = keras.models.load_model(str(best_path)) if best_path.exists() else model
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test Macro F1     : {f1_macro:.4f}")
    print(f"  Test Weighted F1  : {f1_weighted:.4f}")

    # Full classification report
    report = classification_report(
        y_test, y_pred,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0,
    )
    print("\n  Classification Report:")
    print(report)

    report_path = Path(cfg.paths.reports_dir) / "english_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"English CNN-BiLSTM Test Set Evaluation\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n\n")
        f.write(report)
    print(f"  Report saved -> {report_path}")

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    _plot_training_curves(history, cfg.paths.figures_dir)
    _plot_confusion_matrix(y_test, y_pred, le.classes_, cfg.paths.figures_dir,
                           title="English Model - Confusion Matrix")

    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  English training complete in {elapsed/60:.1f} minutes.")
    print(f"  Best model: {cfg.paths.english_best}")
    print(f"{'='*65}\n")

    return {
        "test_accuracy": acc,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "epochs_trained": len(history.history["loss"]),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_training_curves(history, figures_dir: str) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("English CNN-BiLSTM Training Curves", fontsize=14, fontweight="bold")

    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, h["loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, h["val_loss"], "r-s", label="Val Loss", markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, h["accuracy"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(epochs, h["val_accuracy"], "r-s", label="Val Acc", markersize=3)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "english_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves -> {out_path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    figures_dir: str,
    title: str = "Confusion Matrix",
    filename: str = "english_confusion_matrix.png",
) -> None:
    """Plot and save a normalised confusion matrix heatmap."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()

    out_path = figures_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train_english()
    print(f"\nFinal results:")
    print(f"  Test Accuracy : {results['test_accuracy']:.4f}")
    print(f"  Test Macro F1 : {results['test_f1_macro']:.4f}")
    print(f"  Epochs trained: {results['epochs_trained']}")
