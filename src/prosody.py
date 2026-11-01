"""
src/prosody.py
==============
Prosodic feature extraction from raw audio waveforms.

Extracts a fixed-length prosodic feature VECTOR per audio clip.
This vector is later concatenated with the Log-Mel spectrogram frames
before being fed into the CNN-BiLSTM.

Features extracted (12 dimensions):
    [0]  pitch_mean       — mean F0 over voiced frames (Hz)
    [1]  pitch_std        — standard deviation of F0 (Hz)
    [2]  pitch_range      — F0 max minus F0 min (Hz)
    [3]  voiced_fraction  — proportion of frames detected as voiced (0–1)
    [4]  energy_mean      — mean RMS energy
    [5]  energy_std       — standard deviation of RMS energy
    [6]  energy_max       — maximum RMS energy
    [7]  zcr_mean         — mean zero-crossing rate (speaking rate proxy)
    [8]  zcr_std          — std of zero-crossing rate
    [9]  silence_ratio    — fraction of near-silence frames
    [10] pitch_slope      — linear trend coefficient of F0 (intonation)
    [11] energy_slope     — linear trend coefficient of energy (prominence)

NaN Handling:
    All NaN values are explicitly replaced with 0.0.
    This occurs when: pitch detection fails (unvoiced speech), very short
    clips, or heavily noisy audio. The zero replacement is documented
    in the output and must be accounted for in normalization.

Pitch Extraction:
    Primary:  praat-parselmouth (if available) — better voiced/unvoiced detection
    Fallback: librosa.yin() — good alternative, slightly noisier

Public API
----------
extract_prosodic_features(audio, sr, cfg_prosody) -> np.ndarray  shape (12,)
extract_prosodic_batch(audio_list, sr, cfg_prosody) -> np.ndarray shape (N, 12)
PROSODIC_FEATURE_NAMES : list[str]
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# Pitch backend selection — try Parselmouth first, fall back to librosa
try:
    import parselmouth
    _PARSELMOUTH_AVAILABLE = True
except ImportError:
    _PARSELMOUTH_AVAILABLE = False

import librosa


# ---------------------------------------------------------------------------
# Feature name registry (order must match extraction below)
# ---------------------------------------------------------------------------

PROSODIC_FEATURE_NAMES: list[str] = [
    "pitch_mean",        # [0]
    "pitch_std",         # [1]
    "pitch_range",       # [2]
    "voiced_fraction",   # [3]
    "energy_mean",       # [4]
    "energy_std",        # [5]
    "energy_max",        # [6]
    "zcr_mean",          # [7]
    "zcr_std",           # [8]
    "silence_ratio",     # [9]
    "pitch_slope",       # [10]
    "energy_slope",      # [11]
]

N_PROSODIC_FEATURES = len(PROSODIC_FEATURE_NAMES)  # 12


# ---------------------------------------------------------------------------
# Public: extract_prosodic_features
# ---------------------------------------------------------------------------

def extract_prosodic_features(
    audio: np.ndarray,
    sr: int = 16_000,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    frame_length: float = 0.025,
    hop_length_s: float = 0.010,
    silence_threshold: float = 0.01,
) -> np.ndarray:
    """
    Extract a 12-dimensional prosodic feature vector from one audio clip.

    Parameters
    ----------
    audio : np.ndarray
        Float32 mono waveform, shape (n_samples,).
    sr : int
        Sample rate in Hz.
    f0_min : float
        Minimum F0 search range (Hz).
    f0_max : float
        Maximum F0 search range (Hz).
    frame_length : float
        Analysis frame length in seconds.
    hop_length_s : float
        Analysis frame hop in seconds.
    silence_threshold : float
        RMS below this = silence frame.

    Returns
    -------
    np.ndarray
        Shape (12,), dtype float32. NaN-free (zeros where detection failed).
    """
    if audio.ndim != 1 or audio.size == 0:
        warnings.warn(
            "[prosody] Invalid audio shape or empty array. Returning zeros.",
            stacklevel=2,
        )
        return np.zeros(N_PROSODIC_FEATURES, dtype=np.float32)

    hop_samples = int(sr * hop_length_s)
    frame_samples = int(sr * frame_length)

    # --- Pitch (F0) ---
    f0_values, voiced_flags = _extract_pitch(
        audio, sr=sr, f0_min=f0_min, f0_max=f0_max,
        hop_samples=hop_samples, frame_samples=frame_samples,
    )

    # --- Energy (RMS) ---
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_samples,
        hop_length=hop_samples,
    )[0]  # shape: (n_frames,)

    # --- Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_samples,
        hop_length=hop_samples,
    )[0]  # shape: (n_frames,)

    # --- Derived features ---
    voiced_f0 = f0_values[voiced_flags]  # only voiced frames

    # [0] pitch_mean
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # [1] pitch_std
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0

    # [2] pitch_range
    pitch_range = (float(np.max(voiced_f0) - np.min(voiced_f0))
                   if len(voiced_f0) > 1 else 0.0)

    # [3] voiced_fraction
    voiced_fraction = float(np.mean(voiced_flags)) if len(voiced_flags) > 0 else 0.0

    # [4] energy_mean
    energy_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0

    # [5] energy_std
    energy_std = float(np.std(rms)) if len(rms) > 1 else 0.0

    # [6] energy_max
    energy_max = float(np.max(rms)) if len(rms) > 0 else 0.0

    # [7] zcr_mean
    zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

    # [8] zcr_std
    zcr_std = float(np.std(zcr)) if len(zcr) > 1 else 0.0

    # [9] silence_ratio
    silence_frames = np.sum(rms < silence_threshold)
    silence_ratio = float(silence_frames / len(rms)) if len(rms) > 0 else 0.0

    # [10] pitch_slope (linear trend of F0 over time — captures intonation)
    if len(voiced_f0) >= 3:
        voiced_times = np.where(voiced_flags)[0].astype(np.float32)
        # Fit degree-1 polynomial: slope
        coeffs = np.polyfit(voiced_times, voiced_f0, 1)
        pitch_slope = float(coeffs[0])
    else:
        pitch_slope = 0.0

    # [11] energy_slope (linear trend of energy — captures prominence contour)
    if len(rms) >= 3:
        times = np.arange(len(rms), dtype=np.float32)
        coeffs = np.polyfit(times, rms, 1)
        energy_slope = float(coeffs[0])
    else:
        energy_slope = 0.0

    # Assemble vector
    feature_vector = np.array([
        pitch_mean,
        pitch_std,
        pitch_range,
        voiced_fraction,
        energy_mean,
        energy_std,
        energy_max,
        zcr_mean,
        zcr_std,
        silence_ratio,
        pitch_slope,
        energy_slope,
    ], dtype=np.float32)

    # Final NaN/Inf guard — must never enter the network
    feature_vector = _sanitize_vector(feature_vector)

    return feature_vector


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_prosodic_batch(
    audio_list: list[np.ndarray],
    sr: int = 16_000,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    frame_length: float = 0.025,
    hop_length_s: float = 0.010,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract prosodic features for a list of audio clips.

    Parameters
    ----------
    audio_list : list of np.ndarray
        Each element is a float32 mono waveform.
    sr : int
        Sample rate.
    f0_min, f0_max : float
        F0 search range.
    frame_length, hop_length_s : float
        Analysis frame parameters.
    verbose : bool
        Print progress for large batches.

    Returns
    -------
    np.ndarray
        Shape (N, 12), dtype float32.
    """
    results = []
    for i, audio in enumerate(audio_list):
        if verbose and (i % 50 == 0):
            print(f"  [prosody] {i}/{len(audio_list)} clips...", flush=True)
        vec = extract_prosodic_features(
            audio, sr=sr, f0_min=f0_min, f0_max=f0_max,
            frame_length=frame_length, hop_length_s=hop_length_s,
        )
        results.append(vec)
    return np.stack(results, axis=0)  # (N, 12)


# ---------------------------------------------------------------------------
# Internal: pitch extraction with backend fallback
# ---------------------------------------------------------------------------

def _extract_pitch(
    audio: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
    hop_samples: int,
    frame_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-frame F0 values and voiced/unvoiced flags.

    Returns
    -------
    f0_values : np.ndarray
        F0 in Hz per frame (0.0 where unvoiced).
    voiced_flags : np.ndarray of bool
        True where frame is detected as voiced.
    """
    if _PARSELMOUTH_AVAILABLE:
        return _pitch_parselmouth(audio, sr, f0_min, f0_max, hop_samples)
    else:
        return _pitch_librosa_yin(audio, sr, f0_min, f0_max, hop_samples, frame_samples)


def _pitch_parselmouth(
    audio: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
    hop_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use Parselmouth (Praat) for F0 — more accurate voiced/unvoiced detection."""
    try:
        snd = parselmouth.Sound(values=audio.astype(np.float64), sampling_frequency=sr)
        pitch_obj = snd.to_pitch(
            time_step=hop_samples / sr,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        f0_array = pitch_obj.selected_array["frequency"]  # 0.0 = unvoiced
        voiced = f0_array > 0.0
        return f0_array.astype(np.float32), voiced
    except Exception as exc:
        warnings.warn(
            f"[prosody] Parselmouth pitch failed ({exc}). Falling back to librosa.yin.",
            stacklevel=4,
        )
        frame_samples = int(sr * 0.025)
        return _pitch_librosa_yin(audio, sr, f0_min, f0_max, hop_samples, frame_samples)


def _pitch_librosa_yin(
    audio: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
    hop_samples: int,
    frame_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use librosa.yin for F0 — fallback when Parselmouth unavailable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = librosa.yin(
                audio,
                fmin=f0_min,
                fmax=f0_max,
                sr=sr,
                hop_length=hop_samples,
                frame_length=frame_samples,
            )
        # librosa.yin returns f0_min when unvoiced — mark as unvoiced
        voiced = f0 > (f0_min + 1.0)
        f0 = np.where(voiced, f0, 0.0)
        return f0.astype(np.float32), voiced
    except Exception as exc:
        warnings.warn(
            f"[prosody] librosa.yin failed ({exc}). Returning zero pitch.",
            stacklevel=4,
        )
        n_frames = max(1, len(audio) // hop_samples)
        return np.zeros(n_frames, np.float32), np.zeros(n_frames, bool)


# ---------------------------------------------------------------------------
# NaN sanitizer for prosodic vectors
# ---------------------------------------------------------------------------

def _sanitize_vector(vec: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in prosodic vector with 0.0 and warn."""
    bad_mask = ~np.isfinite(vec)
    if bad_mask.any():
        bad_names = [PROSODIC_FEATURE_NAMES[i] for i in np.where(bad_mask)[0]]
        warnings.warn(
            f"[prosody] Non-finite values detected in features: {bad_names}. "
            "Replacing with 0.0. This may indicate: silence, very short clip, "
            "or noisy audio.",
            stacklevel=3,
        )
        vec = np.where(bad_mask, 0.0, vec)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature display helper (used by predict.py and app.py)
# ---------------------------------------------------------------------------

def format_prosodic_features(
    vec: np.ndarray,
    normalized_vec: Optional[np.ndarray] = None,
) -> str:
    """
    Format a prosodic vector as a human-readable string.

    Parameters
    ----------
    vec : np.ndarray
        Raw prosodic feature vector (12,).
    normalized_vec : np.ndarray or None
        Normalized version (12,) — if provided, displayed alongside.

    Returns
    -------
    str
        Formatted multi-line string.
    """
    lines = [f"{'Feature':<20} {'Raw':>10}"]
    if normalized_vec is not None:
        lines[0] += f"  {'Normalized':>12}"
    lines.append("-" * (34 if normalized_vec is None else 46))
    for i, name in enumerate(PROSODIC_FEATURE_NAMES):
        raw_val = f"{vec[i]:10.4f}"
        line = f"  {name:<20} {raw_val}"
        if normalized_vec is not None:
            line += f"  {normalized_vec[i]:12.4f}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Prosody self-test")
    print(f"  Pitch backend: {'parselmouth' if _PARSELMOUTH_AVAILABLE else 'librosa.yin'}")
    print("=" * 50)

    SR = 16_000
    DURATION = 3.0

    # Synthesize a voiced signal (sine wave — should be detected as voiced)
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    vec = extract_prosodic_features(audio, sr=SR)
    assert vec.shape == (N_PROSODIC_FEATURES,), f"Wrong shape: {vec.shape}"
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec)), f"NaN/Inf in output: {vec}"
    print(f"[OK] extract_prosodic_features: shape={vec.shape}")
    print(format_prosodic_features(vec))

    # Test batch
    batch = extract_prosodic_batch([audio, audio], sr=SR, verbose=False)
    assert batch.shape == (2, N_PROSODIC_FEATURES), f"Batch shape: {batch.shape}"
    print(f"\n[OK] extract_prosodic_batch: shape={batch.shape}")

    # Test silence
    silence = np.zeros(int(SR * DURATION), dtype=np.float32)
    vec_s = extract_prosodic_features(silence, sr=SR)
    assert np.all(np.isfinite(vec_s))
    print(f"[OK] Silence input -> all-finite zeros: {vec_s}")

    print("\n[PASS] prosody.py self-test complete.")
