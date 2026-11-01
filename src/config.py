"""
src/config.py
=============
Loads config.yaml and exposes a single Config object accessible
throughout the entire codebase.

Usage:
    from src.config import Config
    cfg = Config()
    print(cfg.audio.sample_rate)   # 16000
    print(cfg.paths.english_tess)  # "data/english/TESS"
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Root of the project (one level above src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _Namespace:
    """Recursively converts a dict into attribute-accessible object."""

    def __init__(self, d: dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _Namespace(v))
            else:
                setattr(self, k, v)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Namespace({self.__dict__})"

    def to_dict(self) -> dict[str, Any]:
        """Convert back to plain dict (for serialization)."""
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, _Namespace) else v
        return out


class Config:
    """
    Singleton-like configuration loader.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config.yaml. Defaults to <project_root>/config.yaml.

    Examples
    --------
    >>> cfg = Config()
    >>> cfg.audio.sample_rate
    16000
    >>> cfg.paths.english_tess
    'data/english/TESS'
    """

    _instance: "Config | None" = None

    def __new__(cls, config_path: str | Path | None = None) -> "Config":  # noqa: F811
        # Allow re-loading if a path is explicitly given, otherwise reuse
        if cls._instance is None or config_path is not None:
            inst = super().__new__(cls)
            inst._loaded = False
            cls._instance = inst
        return cls._instance

    def __init__(self, config_path: str | Path | None = None) -> None:
        if self._loaded:
            return
        path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"[Config] config.yaml not found at: {path}\n"
                "Make sure you are running from the project root directory."
            )
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        # Expose each top-level section as an attribute
        for section, values in raw.items():
            setattr(self, section, _Namespace(values) if isinstance(values, dict) else values)

        # Resolve relative paths to absolute using PROJECT_ROOT
        self._resolve_paths()

        # Set random seeds immediately
        self._set_seeds()

        self._loaded = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> None:
        """Make all path values absolute if they are relative."""
        p = self.paths
        for attr in vars(p):
            val = getattr(p, attr)
            if isinstance(val, str) and not os.path.isabs(val):
                setattr(p, attr, str(PROJECT_ROOT / val))

    def _set_seeds(self) -> None:
        """Fix all random seeds for reproducibility."""
        seed = getattr(self, "seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf  # noqa: PLC0415
            tf.random.set_seed(seed)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """
        Create all output directories referenced in config if they do
        not already exist. Call this once at the start of each script.
        """
        dirs_to_create = [
            self.paths.results_dir,
            self.paths.figures_dir,
            self.paths.reports_dir,
            str(Path(self.paths.english_best).parent),  # models/
        ]
        for d in dirs_to_create:
            Path(d).mkdir(parents=True, exist_ok=True)

    def get_n_classes(self, language: str = "english") -> int:
        """Return the number of emotion classes for the given language."""
        if language.lower() == "english":
            return int(self.emotions.n_classes)
        # For Tamil, use same unless overridden in future
        return int(self.emotions.n_classes)

    def get_input_shape(self) -> tuple[int, int]:
        """
        Return the (time_steps, features) shape of one input sample.

        time_steps  = ceil(duration * sample_rate / hop_length)
        features    = n_mels + n_prosodic_features
        """
        import math
        time_steps = math.ceil(
            self.audio.duration * self.audio.sample_rate / self.features.hop_length
        )
        features = self.features.n_mels + self.prosody.n_prosodic_features
        return (time_steps, features)

    def get_tess_label_map(self) -> dict[str, str]:
        """
        TESS uses 'ps' for pleasant surprise.
        Return normalisation mapping from TESS folder names to canonical
        emotion labels defined in config.yaml.
        """
        return {
            "angry": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "happy",
            "neutral": "neutral",
            "ps": "surprise",   # TESS-specific mapping
            "surprise": "surprise",
            "pleasant_surprise": "surprise",
            "pleasant_surprised": "surprise",
            "sad": "sad",
        }

    def summary(self) -> str:  # pragma: no cover
        """Human-readable config summary."""
        lines = ["=" * 60, "  PN-CLT Configuration Summary", "=" * 60]
        input_shape = self.get_input_shape()
        lines += [
            f"  Sample rate      : {self.audio.sample_rate} Hz",
            f"  Duration         : {self.audio.duration} s",
            f"  Mel bins         : {self.features.n_mels}",
            f"  FFT size         : {self.features.n_fft}",
            f"  Hop length       : {self.features.hop_length}",
            f"  Prosodic dims    : {self.prosody.n_prosodic_features}",
            f"  Input shape      : {input_shape}",
            f"  Emotion classes  : {self.emotions.n_classes}",
            f"  English TESS dir : {self.paths.english_tess}",
            f"  Tamil dir        : {self.paths.tamil_data}",
            f"  Random seed      : {self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience: import cfg from anywhere
# ---------------------------------------------------------------------------
cfg = Config()


if __name__ == "__main__":
    print(cfg.summary())
    cfg.ensure_dirs()
    print("[OK] Config loaded and directories ensured.")
