"""
tests/test_core_pipeline.py
===========================
Comprehensive unit test suite for PN-CLT SER pipeline.
Tests run in seconds using synthetic fixtures without requiring large real datasets.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.preprocessing import (
    load_audio, pad_or_truncate, normalize_amplitude, is_silent, split_indices
)
from src.feature_extraction import (
    extract_log_mel, extract_features, extract_deltas, normalize_spectrogram, get_expected_frames
)
from src.prosody import extract_prosodic_features, N_PROSODIC_FEATURES
from src.normalization import (
    ProsodicNormalizer, append_prosody_to_spectrogram
)
from src.model import (
    build_cnn_bilstm, replace_classification_head, freeze_layers, unfreeze_layers, compile_model,
    FROZEN_CORE_LAYERS, UPPER_LAYERS
)
from src.dataset import scan_tess, scan_tamil, combine_features


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_load_audio_shape_and_type(tmp_wav_file: Path, sample_rate: int):
    audio = load_audio(tmp_wav_file, sr=sample_rate, duration=3.0)
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (sample_rate * 3,)
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6


def test_load_audio_stereo_to_mono(tmp_multi_channel_wav_file: Path, sample_rate: int):
    audio = load_audio(tmp_multi_channel_wav_file, sr=sample_rate, duration=3.0)
    assert audio.ndim == 1
    assert audio.shape == (sample_rate * 3,)


def test_pad_or_truncate_short(sine_audio: np.ndarray):
    short = sine_audio[:1000]
    target_len = 48000
    padded = pad_or_truncate(short, target_len)
    assert padded.shape == (target_len,)
    # Right padding with zeros
    assert np.all(padded[1000:] == 0.0)


def test_pad_or_truncate_long(sine_audio: np.ndarray):
    target_len = 10000
    cropped = pad_or_truncate(sine_audio, target_len)
    assert cropped.shape == (target_len,)


def test_is_silent(silence_audio: np.ndarray, sine_audio: np.ndarray):
    assert is_silent(silence_audio)
    assert not is_silent(sine_audio)


def test_split_indices_proportions():
    n = 100
    train, val, test = split_indices(n, val_frac=0.15, test_frac=0.15, seed=42)
    assert len(train) + len(val) + len(test) == n
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Disjoint sets
    all_idx = set(train) | set(val) | set(test)
    assert len(all_idx) == n


def test_split_indices_small_dataset():
    # Test defensive split on tiny sample sizes
    train, val, test = split_indices(6, val_frac=0.2, test_frac=0.2, seed=42)
    assert len(train) >= 1
    assert len(val) >= 1
    assert len(test) >= 1
    assert len(train) + len(val) + len(test) == 6


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_log_mel_shape(sine_audio: np.ndarray, sample_rate: int):
    n_mels = 128
    hop_length = 512
    mel = extract_log_mel(sine_audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    expected_frames = get_expected_frames(sample_rate, 3.0, hop_length)
    assert mel.shape == (expected_frames, n_mels)
    assert not np.isnan(mel).any()
    assert not np.isinf(mel).any()


def test_extract_features_with_deltas(sine_audio: np.ndarray, sample_rate: int):
    n_mels = 128
    hop = 512
    mel_deltas = extract_features(sine_audio, sr=sample_rate, n_mels=n_mels, hop_length=hop, include_deltas=True)
    expected_frames = get_expected_frames(sample_rate, 3.0, hop)
    assert mel_deltas.shape == (expected_frames, n_mels * 3)


def test_normalize_spectrogram(sine_audio: np.ndarray, sample_rate: int):
    mel = extract_log_mel(sine_audio, sr=sample_rate)
    norm = normalize_spectrogram(mel)
    assert abs(float(norm.mean())) < 0.05
    assert abs(float(norm.std()) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Prosodic Features & Normalization Tests
# ---------------------------------------------------------------------------

def test_extract_prosodic_features_dimension(sine_audio: np.ndarray, sample_rate: int):
    vec = extract_prosodic_features(sine_audio, sr=sample_rate)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (N_PROSODIC_FEATURES,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_prosodic_normalizer_fit_transform_save_load():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=15.0, size=(40, 12)).astype(np.float32)

    norm = ProsodicNormalizer(language="tamil", method="zscore")
    norm.fit(data)
    transformed = norm.transform(data)

    assert transformed.shape == (40, 12)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "test_norm.json"
        norm.save(json_path)
        assert json_path.exists()

        loaded_norm = ProsodicNormalizer.load(json_path)
        test_sample = data[:2]
        assert np.allclose(norm.transform(test_sample), loaded_norm.transform(test_sample), atol=1e-6)


def test_append_prosody_to_spectrogram():
    frames = 94
    n_mels = 128
    n_pros = 12
    spec = np.zeros((frames, n_mels), dtype=np.float32)
    pvec = np.ones((n_pros,), dtype=np.float32) * 2.5

    combined = append_prosody_to_spectrogram(spec, pvec)
    assert combined.shape == (frames, n_mels + n_pros)
    assert np.all(combined[:, :n_mels] == 0.0)
    assert np.all(combined[:, n_mels:] == 2.5)


# ---------------------------------------------------------------------------
# Model Architecture & Transfer Learning Tests
# ---------------------------------------------------------------------------

def test_model_build_compile_and_predict():
    input_shape = (94, 140)
    n_classes = 7
    model = build_cnn_bilstm(input_shape=input_shape, n_classes=n_classes)
    compile_model(model, lr=0.001, n_classes=n_classes)

    batch_x = np.random.randn(2, 94, 140).astype(np.float32)
    preds = model.predict(batch_x, verbose=0)
    assert preds.shape == (2, n_classes)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)


def test_model_head_replacement_and_freeze():
    input_shape = (94, 140)
    orig_classes = 7
    new_classes = 5

    model = build_cnn_bilstm(input_shape=input_shape, n_classes=orig_classes)
    freeze_layers(model, FROZEN_CORE_LAYERS)

    # Check that conv1d block is frozen
    conv_layer = model.get_layer("cnn_block1_conv")
    assert not conv_layer.trainable

    # Replace head
    new_model = replace_classification_head(model, n_classes=new_classes)
    assert new_model.output_shape == (None, new_classes)

    # Unfreeze upper layers
    unfreeze_layers(new_model, UPPER_LAYERS)
    bilstm_layer = new_model.get_layer("bilstm")
    assert bilstm_layer.trainable


# ---------------------------------------------------------------------------
# Dataset Scanning Tests
# ---------------------------------------------------------------------------

def test_scan_english_tess(tmp_english_tess_dir: Path):
    cfg = Config()
    label_map = cfg.get_tess_label_map()
    samples = scan_tess(tmp_english_tess_dir, label_map)
    assert len(samples) > 0
    emotions = set(e for _, e in samples)
    # Check that surprise (from ps) is mapped
    assert "surprise" in emotions
    assert "happy" in emotions


def test_scan_tamil(tmp_tamil_wav_dir: Path):
    cfg = Config()
    label_map = {k: v for k, v in vars(cfg.emotions.tamil_label_map).items()}
    samples = scan_tamil(tmp_tamil_wav_dir, label_map)
    assert len(samples) == 5
    for _, emo in samples:
        assert emo != "unknown"
