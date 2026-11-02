"""
src/preprocessing.py
=====================
Audio loading, mono conversion, resampling, amplitude normalization,
silence handling, and fixed-duration padding / truncation.

All functions are stateless pure functions — they accept numpy arrays
or file paths and return numpy arrays. No global state.

Public API
----------
load_audio(path, sr, duration, normalize) -> np.ndarray
pad_or_truncate(audio, target_len) -> np.ndarray
is_silent(audio, threshold) -> bool
batch_preprocess(paths, sr, duration, normalize) -> list[np.ndarray]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants (overridden by config at call-site)
# ---------------------------------------------------------------------------
DEFAULT_SR = 16_000
DEFAULT_DURATION = 3.0          # seconds
SILENCE_THRESHOLD = 1e-4        # RMS threshold below which audio = silent


# ---------------------------------------------------------------------------
# Helper: Decode audio with FFmpeg (M4A, AAC, MP4, etc.)
# ---------------------------------------------------------------------------

def _decode_with_ffmpeg(path: Path | str, target_sr: int = DEFAULT_SR) -> Optional[np.ndarray]:
    """
    Decode audio file using imageio_ffmpeg or system ffmpeg to float32 mono.
    Handles M4A, AAC, MP3, MP4, WMA, and other formats natively.
    """
    import subprocess
    import shutil

    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = shutil.which("ffmpeg")

    if not ffmpeg_exe:
        return None

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", str(path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            return None
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: load_audio
# ---------------------------------------------------------------------------


def load_audio(
    path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    normalize: bool = True,
    allow_silent: bool = True,
) -> np.ndarray:
    """
    Load an audio file, convert to mono, resample, pad/truncate to a
    fixed length, and optionally normalize amplitude.

    Parameters
    ----------
    path : str or Path
        Path to the audio file (WAV, MP3, FLAC, etc.).
    sr : int
        Target sample rate in Hz (default: 16000).
    duration : float
        Fixed clip duration in seconds (default: 3.0).
    normalize : bool
        If True, peak-normalize the waveform to [-1, 1].
    allow_silent : bool
        If False, raise ValueError for near-silent audio.

    Returns
    -------
    np.ndarray
        Float32 mono waveform of shape (sr * duration,).

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the file cannot be decoded, is empty, or is silent
        (when allow_silent=False).
    RuntimeError
        If audio has unexpected shape after loading.
    """
    path = Path(path)

    # --- Existence check ---
    if not path.exists():
        raise FileNotFoundError(
            f"[preprocessing] Audio file not found: {path}\n"
            "Please check the path and try again."
        )

    # --- Load with librosa or ffmpeg (seamlessly handles M4A, AAC, WAV, MP3, etc.) ---
    audio = None
    orig_sr = None

    # Try librosa first (fast for native formats like WAV)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, orig_sr = librosa.load(str(path), sr=None, mono=False)
    except Exception:
        # If librosa fails (common on Windows for M4A/AAC), decode using ffmpeg
        audio = _decode_with_ffmpeg(path, target_sr=sr)
        if audio is not None and audio.size > 0:
            orig_sr = sr

    # If still None, try ffmpeg explicitly before failing
    if audio is None:
        audio = _decode_with_ffmpeg(path, target_sr=sr)
        if audio is not None and audio.size > 0:
            orig_sr = sr

    if audio is None:
        raise ValueError(
            f"[preprocessing] Could not read audio file: {path}\n"
            "  Supported formats: WAV, M4A, MP3, AAC, FLAC, OGG, AIFF."
        )

    # --- Shape check ---
    if audio is None or audio.size == 0:
        raise ValueError(
            f"[preprocessing] Audio file is empty or unreadable: {path}"
        )

    # --- Mono conversion ---
    if audio.ndim == 2:
        # librosa returns (channels, samples) for multi-channel
        audio = librosa.to_mono(audio)
    elif audio.ndim != 1:
        raise RuntimeError(
            f"[preprocessing] Unexpected audio shape {audio.shape} in: {path}"
        )

    audio = audio.astype(np.float32)

    # --- Resample ---
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # --- Silence detection ---
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < SILENCE_THRESHOLD:
        if not allow_silent:
            raise ValueError(
                f"[preprocessing] Audio appears to be silent (RMS={rms:.2e}): {path}\n"
                "  Check that the file contains actual speech."
            )
        else:
            warnings.warn(
                f"[preprocessing] Near-silent audio detected (RMS={rms:.2e}): {path}",
                stacklevel=2,
            )

    # --- Very short clip warning ---
    min_samples = int(sr * 0.05)   # 50 ms minimum
    if audio.size < min_samples:
        warnings.warn(
            f"[preprocessing] Very short audio ({audio.size/sr:.3f}s < 0.05s): {path}. "
            "Will be zero-padded.",
            stacklevel=2,
        )

    # --- Pad / Truncate ---
    target_len = int(sr * duration)
    audio = pad_or_truncate(audio, target_len)

    # --- Amplitude normalization ---
    if normalize:
        audio = normalize_amplitude(audio)

    return audio


# ---------------------------------------------------------------------------
# Pad / Truncate
# ---------------------------------------------------------------------------

def pad_or_truncate(audio: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ensure audio has exactly `target_len` samples.

    - If shorter: zero-pad at the end (right padding).
    - If longer: truncate from the beginning (take last target_len samples
      to capture the "core" speech, which tends to appear in the middle
      of most emotional speech clips).

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 audio array.
    target_len : int
        Required number of samples.

    Returns
    -------
    np.ndarray
        Array of exactly shape (target_len,).
    """
    current_len = len(audio)
    if current_len == target_len:
        return audio.copy()
    if current_len < target_len:
        pad_width = target_len - current_len
        return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)
    # Truncate — center crop to keep the most emotionally expressive region
    start = (current_len - target_len) // 2
    return audio[start : start + target_len].copy()


# ---------------------------------------------------------------------------
# Amplitude Normalization
# ---------------------------------------------------------------------------

def normalize_amplitude(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1].

    Handles silence gracefully (returns zeros, not NaN).

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio waveform.
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    np.ndarray
        Normalized float32 waveform.
    """
    peak = np.max(np.abs(audio))
    if peak < eps:
        return audio.copy()   # already near-zero — don't distort
    return (audio / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Silence Detection
# ---------------------------------------------------------------------------

def is_silent(
    audio: np.ndarray,
    threshold: float = SILENCE_THRESHOLD,
) -> bool:
    """
    Return True if the RMS energy of the audio is below `threshold`.

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio waveform.
    threshold : float
        RMS threshold below which audio is considered silent.
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < threshold


# ---------------------------------------------------------------------------
# Batch Preprocessing
# ---------------------------------------------------------------------------

def batch_preprocess(
    paths: list[str | Path],
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    normalize: bool = True,
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    Load and preprocess a list of audio files.

    Skips corrupted/missing files with a warning rather than crashing
    the entire batch.

    Parameters
    ----------
    paths : list of str or Path
        Audio file paths.
    sr : int
        Target sample rate.
    duration : float
        Fixed clip duration.
    normalize : bool
        If True, peak-normalize each clip.
    verbose : bool
        Print progress for large batches.

    Returns
    -------
    list of np.ndarray
        Successfully loaded and preprocessed audio arrays.
        Length may be shorter than `paths` if some files fail.
    """
    results: list[np.ndarray] = []
    failed: list[str] = []

    for i, p in enumerate(paths):
        if verbose and (i % 100 == 0):
            print(f"  [batch_preprocess] {i}/{len(paths)} files processed...", flush=True)
        try:
            audio = load_audio(p, sr=sr, duration=duration, normalize=normalize)
            results.append(audio)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            warnings.warn(f"[batch_preprocess] Skipping {p}: {exc}", stacklevel=2)
            failed.append(str(p))

    if failed:
        print(
            f"\n[batch_preprocess] WARNING: {len(failed)} file(s) could not be loaded:\n"
            + "\n".join(f"  • {f}" for f in failed[:10])
            + ("\n  ... (truncated)" if len(failed) > 10 else "")
        )

    return results


# ---------------------------------------------------------------------------
# Train / Validation / Test Split (index-based, stratified)
# ---------------------------------------------------------------------------

def split_indices(
    n: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return shuffled train / val / test index arrays.

    Parameters
    ----------
    n : int
        Total number of samples.
    val_frac : float
        Fraction for validation (0–1).
    test_frac : float
        Fraction for test (0–1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    if n <= 1:
        return idx, np.array([], dtype=int), np.array([], dtype=int)
    if n == 2:
        return idx[:1], idx[1:2], np.array([], dtype=int)

    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        n_test = max(1, (n - 1) // 3)
        n_val = max(1, (n - 1) // 3)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import soundfile as sf

    print("=" * 50)
    print("  Preprocessing self-test")
    print("=" * 50)

    # Create a synthetic 5-second sine wave
    sr = 16_000
    t = np.linspace(0, 5.0, sr * 5)
    sine = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, sine, sr)
        tmp_path = f.name

    audio = load_audio(tmp_path, sr=sr, duration=3.0)
    assert audio.shape == (sr * 3,), f"Shape mismatch: {audio.shape}"
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6
    print(f"[OK] load_audio: shape={audio.shape}, max={np.max(np.abs(audio)):.4f}")

    # Test silence detection
    silence = np.zeros(sr * 3, dtype=np.float32)
    assert is_silent(silence)
    assert not is_silent(audio)
    print("[OK] is_silent")

    # Test split
    train, val, test = split_indices(100, 0.15, 0.15, seed=42)
    assert len(train) + len(val) + len(test) == 100
    print(f"[OK] split_indices: train={len(train)}, val={len(val)}, test={len(test)}")

    import os
    os.unlink(tmp_path)
    print("\n[PASS] preprocessing.py self-test complete.")
