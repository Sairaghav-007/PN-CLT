"""
tests/conftest.py
=================
Shared pytest fixtures used across all test modules.
Generates synthetic audio signals so tests run without any real dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import tempfile
import os
from pathlib import Path


SR = 16000           # sample rate used throughout tests
DURATION = 3.0       # seconds
N_SAMPLES = int(SR * DURATION)


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

def make_sine(freq: float = 220.0, sr: int = SR, duration: float = DURATION,
              amplitude: float = 0.5) -> np.ndarray:
    """Pure sine wave — deterministic, no dataset required."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_noise(sr: int = SR, duration: float = DURATION, seed: int = 0) -> np.ndarray:
    """White noise — deterministic with fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, int(sr * duration)).astype(np.float32)


def make_silence(sr: int = SR, duration: float = DURATION) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_rate() -> int:
    return SR


@pytest.fixture(scope="session")
def sine_audio() -> np.ndarray:
    """A clean 220 Hz sine wave (3 seconds, 16 kHz)."""
    return make_sine()


@pytest.fixture(scope="session")
def noise_audio() -> np.ndarray:
    """White noise audio."""
    return make_noise()


@pytest.fixture(scope="session")
def silence_audio() -> np.ndarray:
    return make_silence()


@pytest.fixture
def tmp_wav_file(tmp_path: Path) -> Path:
    """Write a sine wave to a temporary WAV file and return its path."""
    audio = make_sine()
    fpath = tmp_path / "test_audio.wav"
    sf.write(str(fpath), audio, SR)
    return fpath


@pytest.fixture
def tmp_short_wav_file(tmp_path: Path) -> Path:
    """Very short audio (0.1 seconds) — tests padding logic."""
    audio = make_sine(duration=0.1)
    fpath = tmp_path / "short_audio.wav"
    sf.write(str(fpath), audio, SR)
    return fpath


@pytest.fixture
def tmp_silence_wav_file(tmp_path: Path) -> Path:
    """Silence WAV file."""
    audio = make_silence()
    fpath = tmp_path / "silence_audio.wav"
    sf.write(str(fpath), audio, SR)
    return fpath


@pytest.fixture
def tmp_multi_channel_wav_file(tmp_path: Path) -> Path:
    """Stereo WAV to test mono conversion."""
    mono = make_sine()
    stereo = np.stack([mono, mono * 0.9], axis=1)
    fpath = tmp_path / "stereo_audio.wav"
    sf.write(str(fpath), stereo, SR)
    return fpath


@pytest.fixture
def nonexistent_path(tmp_path: Path) -> Path:
    return tmp_path / "does_not_exist.wav"


@pytest.fixture
def tmp_tamil_wav_dir(tmp_path: Path) -> Path:
    """
    A temporary directory with 5 synthetic Tamil-like WAV files.
    File names encode emotion labels (for dataset loading tests).
    """
    emotions = ["happy", "sad", "angry", "neutral", "fear"]
    tamil_dir = tmp_path / "tamil"
    tamil_dir.mkdir()
    for i, emo in enumerate(emotions):
        audio = make_sine(freq=150.0 + i * 30, duration=2.5)
        fpath = tamil_dir / f"tamil_{emo}_0{i}.wav"
        sf.write(str(fpath), audio, SR)
    return tamil_dir


@pytest.fixture
def tmp_english_tess_dir(tmp_path: Path) -> Path:
    """
    Mimics TESS folder structure:
      TESS/OAF_angry/OAF_angry_01.wav
      TESS/OAF_happy/OAF_happy_01.wav
    """
    tess_dir = tmp_path / "TESS"
    emotions = ["angry", "happy", "sad", "neutral", "fear", "disgust", "ps"]
    speakers = ["OAF", "YAF"]
    for spk in speakers:
        for emo in emotions:
            folder = tess_dir / f"{spk}_{emo}"
            folder.mkdir(parents=True)
            for i in range(3):
                audio = make_sine(freq=200.0, duration=2.0)
                sf.write(str(folder / f"{spk}_{emo}_{i:02d}.wav"), audio, SR)
    return tess_dir
