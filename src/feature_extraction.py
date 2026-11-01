"""
src/feature_extraction.py
===========================
Spectral feature extraction: Log-Mel Spectrogram and MFSC.

IMPORTANT TERMINOLOGY NOTE:
  - MFCC  = Mel-Frequency Cepstral Coefficients (uses DCT — NOT used here)
  - MFSC  = Mel-Frequency Spectral Coefficients = Log-Mel Spectrogram values
             (no DCT step; preserves full spectral envelope)
  - These two are NOT the same. This module computes MFSC / Log-Mel.

The CNN learns spatial patterns directly from the log-energy spectrogram,
which retains more spectral detail than MFCC.

Public API
----------
extract_log_mel(audio, sr, n_mels, n_fft, hop_length, fmin, fmax)
    -> np.ndarray  shape (time_frames, n_mels)

extract_features(audio, sr, cfg)
    -> np.ndarray  shape (time_frames, n_mels)   [spectral only]

The prosodic feature vector is handled separately in prosody.py and
concatenated in dataset.py before feeding into the model.
"""

from __future__ import annotations

import warnings
from typing import Optional

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Core: Log-Mel / MFSC extraction
# ---------------------------------------------------------------------------

def extract_log_mel(
    audio: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
    use_log: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Compute the Log-Mel Spectrogram (MFSC) from a raw waveform.

    This is the primary spectral representation fed to the CNN.

    Parameters
    ----------
    audio : np.ndarray
        Float32 mono waveform, shape (n_samples,).
    sr : int
        Sample rate in Hz.
    n_mels : int
        Number of Mel filter banks (frequency bins).
    n_fft : int
        FFT window size in samples.
    hop_length : int
        Number of samples between successive frames.
    fmin : float
        Lowest frequency for Mel filter bank (Hz).
    fmax : float or None
        Highest frequency for Mel filter bank (Hz).
        If None, uses sr / 2.
    use_log : bool
        If True, apply log(mel + eps) → MFSC representation.
        If False, return raw power Mel spectrogram.
    eps : float
        Small constant for numerical stability in log.

    Returns
    -------
    np.ndarray
        Shape (time_frames, n_mels), dtype float32.
        time_frames = ceil(n_samples / hop_length)

    Raises
    ------
    ValueError
        If audio is empty or not 1-D.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"[feature_extraction] Expected 1-D audio, got shape {audio.shape}."
        )
    if audio.size == 0:
        raise ValueError("[feature_extraction] Audio array is empty.")

    # Compute Mel spectrogram (power, not dB)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,      # power spectrogram (amplitude squared)
    )
    # mel_spec shape: (n_mels, time_frames)

    if use_log:
        # Convert to log scale (MFSC)
        # Using natural log; equivalent to librosa.power_to_db but more direct
        log_mel = np.log(mel_spec + eps)
    else:
        log_mel = mel_spec

    # Transpose to (time_frames, n_mels) — CNN expects (time, features)
    log_mel = log_mel.T.astype(np.float32)

    # Sanity: replace any remaining NaN/Inf
    log_mel = _sanitize(log_mel, name="log_mel")

    return log_mel


# ---------------------------------------------------------------------------
# Delta and Delta-Delta (optional enhancement)
# ---------------------------------------------------------------------------

def extract_deltas(
    spectrogram: np.ndarray,
    width: int = 9,
) -> np.ndarray:
    """
    Compute first and second order temporal derivatives (delta features).

    These capture the rate of change of spectral features over time,
    giving the network information about spectral dynamics.

    Parameters
    ----------
    spectrogram : np.ndarray
        Shape (time_frames, features).
    width : int
        Number of frames used for delta computation (must be odd).

    Returns
    -------
    np.ndarray
        Shape (time_frames, features * 3) — original + delta + delta-delta.
    """
    # librosa expects (features, time) — transpose in and out
    spec_T = spectrogram.T  # (features, time_frames)
    delta1 = librosa.feature.delta(spec_T, width=width, order=1)
    delta2 = librosa.feature.delta(spec_T, width=width, order=2)
    # Stack along feature axis: (features*3, time_frames)
    stacked = np.concatenate([spec_T, delta1, delta2], axis=0)
    return stacked.T.astype(np.float32)  # (time_frames, features*3)


# ---------------------------------------------------------------------------
# High-level extractor (reads from config)
# ---------------------------------------------------------------------------

def extract_features(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    use_log: bool = True,
    include_deltas: bool = False,
) -> np.ndarray:
    """
    High-level feature extractor: Log-Mel (+ optional deltas).

    Parameters
    ----------
    audio : np.ndarray
        Preprocessed float32 mono waveform.
    sr : int
        Sample rate.
    n_mels, n_fft, hop_length, fmin, fmax, use_log :
        Passed directly to extract_log_mel.
    include_deltas : bool
        If True, append delta and delta-delta features.

    Returns
    -------
    np.ndarray
        Shape (time_frames, n_mels) or (time_frames, n_mels * 3) with deltas.
    """
    log_mel = extract_log_mel(
        audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        use_log=use_log,
    )

    if include_deltas:
        log_mel = extract_deltas(log_mel)

    return log_mel


# ---------------------------------------------------------------------------
# Normalization of spectrogram (per-file mean-variance)
# ---------------------------------------------------------------------------

def normalize_spectrogram(
    spectrogram: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize spectrogram to zero mean and unit variance (per sample).

    This is applied PER FILE (not globally) to make the CNN input
    invariant to absolute loudness levels.

    Parameters
    ----------
    spectrogram : np.ndarray
        Shape (time_frames, features).
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    np.ndarray
        Normalized spectrogram, same shape.
    """
    mean = spectrogram.mean()
    std = spectrogram.std()
    return ((spectrogram - mean) / (std + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sanitize(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Replace NaN and Inf with 0.0 and warn."""
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    if n_nan > 0 or n_inf > 0:
        warnings.warn(
            f"[feature_extraction] {name} has {n_nan} NaN and {n_inf} Inf values. "
            "Replacing with 0.0.",
            stacklevel=3,
        )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def get_expected_frames(sr: int, duration: float, hop_length: int) -> int:
    """
    Calculate expected number of time frames for a given duration.

    Used to verify tensor shapes before feeding into the model.

    Parameters
    ----------
    sr : int
        Sample rate.
    duration : float
        Clip duration in seconds.
    hop_length : int
        STFT hop length.

    Returns
    -------
    int
        Expected number of spectrogram time frames.
    """
    import math
    n_samples = int(sr * duration)
    return math.ceil(n_samples / hop_length)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print("=" * 50)
    print("  Feature Extraction self-test")
    print("=" * 50)

    SR = 16_000
    DURATION = 3.0
    N_MELS = 128
    HOP = 512

    # Synthetic sine wave
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    audio = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # Extract log-mel
    log_mel = extract_log_mel(audio, sr=SR, n_mels=N_MELS, hop_length=HOP)
    expected_frames = math.ceil(len(audio) / HOP)
    assert log_mel.shape == (expected_frames, N_MELS), (
        f"Shape mismatch: got {log_mel.shape}, expected ({expected_frames},{N_MELS})"
    )
    assert log_mel.dtype == np.float32
    assert not np.any(np.isnan(log_mel)), "NaN in log-mel output"
    assert not np.any(np.isinf(log_mel)), "Inf in log-mel output"
    print(f"[OK] extract_log_mel: shape={log_mel.shape}, "
          f"min={log_mel.min():.2f}, max={log_mel.max():.2f}")

    # Test with deltas
    log_mel_d = extract_features(audio, SR, include_deltas=True)
    assert log_mel_d.shape == (expected_frames, N_MELS * 3), (
        f"Delta shape mismatch: {log_mel_d.shape}"
    )
    print(f"[OK] extract_features (with deltas): shape={log_mel_d.shape}")

    # Test normalization
    norm = normalize_spectrogram(log_mel)
    assert abs(norm.mean()) < 0.1, f"Mean after norm: {norm.mean():.4f}"
    print(f"[OK] normalize_spectrogram: mean~={norm.mean():.4f}, std~={norm.std():.4f}")

    # Test get_expected_frames
    n = get_expected_frames(SR, DURATION, HOP)
    assert n == expected_frames
    print(f"[OK] get_expected_frames: {n}")

    # Test sanitize — inject NaN
    bad = log_mel.copy()
    bad[0, 0] = float("nan")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clean = _sanitize(bad, name="test")
        assert len(w) == 1 and "NaN" in str(w[0].message)
    assert not np.isnan(clean).any()
    print("[OK] _sanitize: NaN replaced with 0.0")

    print("\n[PASS] feature_extraction.py self-test complete.")
