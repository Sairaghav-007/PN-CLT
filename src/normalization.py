"""
src/normalization.py
=====================
Language-adaptive prosodic normalization.

This is the CENTRAL COMPONENT of the PN-CLT method.

MOTIVATION:
-----------
A Tamil speaker's pitch of 180 Hz may represent mild surprise relative
to Tamil speaking norms, while the same 180 Hz would represent a
normal neutral pitch for an English male speaker.

By z-scoring prosodic features relative to the LANGUAGE-SPECIFIC
baseline (μ and σ estimated from that language's calibration data),
we make the prosodic signal language-invariant:

    z = (x - μ_language) / (σ_language + ε)

The normalized z-value now represents EMOTIONAL DEVIATION FROM THE
LANGUAGE BASELINE, not absolute acoustic values.

This is the key hypothesis of PN-CLT:
"Language-adaptive prosodic normalization improves cross-lingual
 emotion recognition by aligning the emotional prosodic signal
 across source and target languages."

CRITICAL DATA-LEAKAGE RULES:
------------------------------
1. .fit() must only be called on TRAINING / CALIBRATION data.
2. .transform() is applied to validation, test, and inference data.
3. Statistics MUST be saved after fitting and MUST be loaded before
   inference — never recomputed on test data.

SMALL-SAMPLE WARNING:
---------------------
Reliable language-level statistics require at least 30+ diverse samples.
With fewer than 30 samples, normalization statistics are unreliable.
With 1–10 samples (demo / proof-of-concept), the normalizer will
run but MUST display a clear warning that statistics are not valid.

Public API
----------
ProsodicNormalizer
    .fit(X)              — compute μ and σ from calibration data
    .transform(X)        — apply z-score normalization
    .fit_transform(X)    — fit + transform in one call
    .save(path)          — serialize to disk (JSON)
    .load(path)          — deserialize from disk (class method)
    .summary()           — human-readable statistics summary
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SAMPLES_FOR_RELIABLE_STATS = 30


# ---------------------------------------------------------------------------
# ProsodicNormalizer
# ---------------------------------------------------------------------------

class ProsodicNormalizer:
    """
    Language-adaptive prosodic normalizer using z-score normalization.

    Fits per-feature mean and standard deviation on calibration data
    from a specific language, then transforms prosodic vectors so that
    values are expressed as standard deviations from the language baseline.

    Parameters
    ----------
    language : str
        Language identifier (e.g., "english", "tamil"). Stored for
        documentation and metadata purposes.
    method : str
        Normalization method: "zscore" | "minmax" | "robust".
        Default: "zscore".
    epsilon : float
        Small constant added to denominator to prevent division by zero.
        Default: 1e-8.

    Examples
    --------
    >>> norm = ProsodicNormalizer(language="tamil")
    >>> norm.fit(tamil_calibration_features)   # shape (N, 12)
    >>> normalized = norm.transform(tamil_test_features)
    >>> norm.save("models/prosodic_normalizer_tamil.json")

    >>> norm2 = ProsodicNormalizer.load("models/prosodic_normalizer_tamil.json")
    >>> normalized = norm2.transform(new_features)
    """

    def __init__(
        self,
        language: str = "unknown",
        method: str = "zscore",
        epsilon: float = 1e-8,
    ) -> None:
        self.language = language
        self.method = method.lower()
        self.epsilon = epsilon

        # Statistics — set by .fit()
        self.mean_: Optional[np.ndarray] = None    # shape (n_features,)
        self.std_: Optional[np.ndarray] = None     # shape (n_features,)
        self.min_: Optional[np.ndarray] = None     # for minmax
        self.max_: Optional[np.ndarray] = None     # for minmax
        self.median_: Optional[np.ndarray] = None  # for robust
        self.iqr_: Optional[np.ndarray] = None     # for robust
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.feature_names_: list[str] = []
        self._is_fitted: bool = False

        if self.method not in ("zscore", "minmax", "robust"):
            raise ValueError(
                f"[ProsodicNormalizer] Unknown method: '{method}'. "
                "Choose from: 'zscore', 'minmax', 'robust'."
            )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "ProsodicNormalizer":
        """
        Compute normalization statistics from calibration data.

        IMPORTANT: Only call this on TRAINING / CALIBRATION data,
        never on validation or test data (data leakage).

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features). Prosodic feature matrix.
        feature_names : list of str, optional
            Names for each feature column (for display/debugging).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(
                f"[ProsodicNormalizer.fit] Expected 2D array, got shape {X.shape}."
            )

        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self.feature_names_ = (
            feature_names
            or [f"feature_{i}" for i in range(n_features)]
        )

        # --- Small-sample warning ---
        if n_samples < MIN_SAMPLES_FOR_RELIABLE_STATS:
            warnings.warn(
                f"\n{'='*65}\n"
                f"[ProsodicNormalizer] WARNING: Only {n_samples} sample(s) "
                f"for language '{self.language}'.\n"
                f"Reliable normalization statistics require at least "
                f"{MIN_SAMPLES_FOR_RELIABLE_STATS} samples.\n"
                f"With {n_samples} sample(s):\n"
                f"  • Statistics are not statistically meaningful.\n"
                f"  • Results are PROOF-OF-CONCEPT only.\n"
                f"  • Do NOT report normalized values as validated research.\n"
                f"{'='*65}",
                stacklevel=2,
            )

        # Check for NaN in calibration data
        n_nan = np.isnan(X).sum()
        if n_nan > 0:
            warnings.warn(
                f"[ProsodicNormalizer.fit] Input has {n_nan} NaN values. "
                "They will be excluded from statistics computation per feature.",
                stacklevel=2,
            )

        if self.method == "zscore":
            # NaN-safe mean and std (ignore NaN entries per feature)
            self.mean_ = np.nanmean(X, axis=0).astype(np.float32)
            self.std_ = np.nanstd(X, axis=0).astype(np.float32)

        elif self.method == "minmax":
            self.min_ = np.nanmin(X, axis=0).astype(np.float32)
            self.max_ = np.nanmax(X, axis=0).astype(np.float32)

        elif self.method == "robust":
            self.median_ = np.nanmedian(X, axis=0).astype(np.float32)
            q25 = np.nanpercentile(X, 25, axis=0).astype(np.float32)
            q75 = np.nanpercentile(X, 75, axis=0).astype(np.float32)
            self.iqr_ = (q75 - q25).astype(np.float32)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply language-adaptive normalization to prosodic features.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features) or (n_features,) for a single sample.

        Returns
        -------
        np.ndarray
            Normalized array, same shape as X, dtype float32.

        Raises
        ------
        RuntimeError
            If called before .fit().
        ValueError
            If feature dimension doesn't match fitted dimension.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "[ProsodicNormalizer.transform] Normalizer has not been fitted. "
                "Call .fit(calibration_data) first, or .load(path) to load "
                "pre-fitted statistics."
            )

        X = np.asarray(X, dtype=np.float32)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"[ProsodicNormalizer.transform] Feature dimension mismatch. "
                f"Expected {self.n_features_}, got {X.shape[1]}."
            )

        X_norm = X.copy()

        if self.method == "zscore":
            X_norm = (X - self.mean_) / (self.std_ + self.epsilon)

        elif self.method == "minmax":
            denom = (self.max_ - self.min_) + self.epsilon
            X_norm = (X - self.min_) / denom

        elif self.method == "robust":
            X_norm = (X - self.median_) / (self.iqr_ + self.epsilon)

        # Final NaN/Inf guard — std=0 features become NaN without epsilon,
        # but with epsilon they should be 0. Guard anyway.
        bad = ~np.isfinite(X_norm)
        if bad.any():
            warnings.warn(
                f"[ProsodicNormalizer.transform] {bad.sum()} non-finite values "
                "after normalization. Replacing with 0.0.",
                stacklevel=2,
            )
            X_norm = np.where(bad, 0.0, X_norm)

        return X_norm[0] if single else X_norm.astype(np.float32)

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Fit on X and return normalized X. Equivalent to fit(X).transform(X)."""
        return self.fit(X, feature_names=feature_names).transform(X)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save normalizer statistics to a JSON file.

        JSON is used (over pickle) for transparency and
        cross-platform compatibility.

        Parameters
        ----------
        path : str or Path
            Output path. Parent directories created if needed.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "[ProsodicNormalizer.save] Cannot save — normalizer not fitted."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "language": self.language,
            "method": self.method,
            "epsilon": float(self.epsilon),
            "n_samples": int(self.n_samples_),
            "n_features": int(self.n_features_),
            "feature_names": self.feature_names_,
            "reliable": self.n_samples_ >= MIN_SAMPLES_FOR_RELIABLE_STATS,
            "min_samples_for_reliability": MIN_SAMPLES_FOR_RELIABLE_STATS,
        }

        def _arr(a: Optional[np.ndarray]) -> Optional[list]:
            return a.tolist() if a is not None else None

        data["mean"] = _arr(self.mean_)
        data["std"] = _arr(self.std_)
        data["min"] = _arr(self.min_)
        data["max"] = _arr(self.max_)
        data["median"] = _arr(self.median_)
        data["iqr"] = _arr(self.iqr_)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[ProsodicNormalizer] Saved statistics to: {path}")
        if not data["reliable"]:
            print(
                f"  ⚠️  WARNING: Only {self.n_samples_} samples used for fitting. "
                "Statistics are NOT statistically reliable."
            )

    @classmethod
    def load(cls, path: str | Path) -> "ProsodicNormalizer":
        """
        Load a previously saved ProsodicNormalizer from JSON.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file created by .save().

        Returns
        -------
        ProsodicNormalizer
            Fully fitted normalizer ready for .transform().

        Raises
        ------
        FileNotFoundError
            If the JSON file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"[ProsodicNormalizer.load] File not found: {path}\n"
                "You must train the normalizer first (run adapt_tamil.py)."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        norm = cls(
            language=data["language"],
            method=data["method"],
            epsilon=data["epsilon"],
        )
        norm.n_samples_ = data["n_samples"]
        norm.n_features_ = data["n_features"]
        norm.feature_names_ = data["feature_names"]

        def _load_arr(key: str) -> Optional[np.ndarray]:
            val = data.get(key)
            return np.array(val, dtype=np.float32) if val is not None else None

        norm.mean_ = _load_arr("mean")
        norm.std_ = _load_arr("std")
        norm.min_ = _load_arr("min")
        norm.max_ = _load_arr("max")
        norm.median_ = _load_arr("median")
        norm.iqr_ = _load_arr("iqr")
        norm._is_fitted = True

        if not data.get("reliable", True):
            warnings.warn(
                f"[ProsodicNormalizer.load] Loaded normalizer was fitted on only "
                f"{norm.n_samples_} samples (< {MIN_SAMPLES_FOR_RELIABLE_STATS}). "
                "Statistics are NOT statistically reliable. "
                "Treat results as proof-of-concept only.",
                stacklevel=2,
            )
        return norm

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable statistics summary."""
        if not self._is_fitted:
            return f"ProsodicNormalizer(language={self.language}, NOT FITTED)"

        lines = [
            "=" * 70,
            f"  ProsodicNormalizer | language={self.language} | method={self.method}",
            f"  n_samples={self.n_samples_} | n_features={self.n_features_}",
            f"  Reliable: {self.n_samples_ >= MIN_SAMPLES_FOR_RELIABLE_STATS} "
            f"(needs >= {MIN_SAMPLES_FOR_RELIABLE_STATS} samples)",
            "=" * 70,
        ]

        if self.method == "zscore" and self.mean_ is not None:
            lines.append(f"  {'Feature':<22} {'Mean':>10} {'Std':>10}")
            lines.append("  " + "-" * 44)
            for i, name in enumerate(self.feature_names_):
                lines.append(
                    f"  {name:<22} {self.mean_[i]:10.4f} {self.std_[i]:10.4f}"
                )
        elif self.method == "minmax" and self.min_ is not None:
            lines.append(f"  {'Feature':<22} {'Min':>10} {'Max':>10}")
            lines.append("  " + "-" * 44)
            for i, name in enumerate(self.feature_names_):
                lines.append(
                    f"  {name:<22} {self.min_[i]:10.4f} {self.max_[i]:10.4f}"
                )
        elif self.method == "robust" and self.median_ is not None:
            lines.append(f"  {'Feature':<22} {'Median':>10} {'IQR':>10}")
            lines.append("  " + "-" * 44)
            for i, name in enumerate(self.feature_names_):
                lines.append(
                    f"  {name:<22} {self.median_[i]:10.4f} {self.iqr_[i]:10.4f}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ProsodicNormalizer(language='{self.language}', "
            f"method='{self.method}', {fitted}, "
            f"n_samples={self.n_samples_})"
        )


# ---------------------------------------------------------------------------
# Utility: append prosodic vector to spectrogram frames
# ---------------------------------------------------------------------------

def append_prosody_to_spectrogram(
    spectrogram: np.ndarray,
    prosodic_vec: np.ndarray,
) -> np.ndarray:
    """
    Concatenate a per-clip prosodic vector to every time frame of a spectrogram.

    This is how the prosodic features integrate with the CNN-BiLSTM:
    each time frame receives the SAME prosodic context vector, allowing
    the CNN to jointly learn spectral + prosodic patterns.

    Parameters
    ----------
    spectrogram : np.ndarray
        Shape (time_frames, n_mels).
    prosodic_vec : np.ndarray
        Shape (n_prosodic,).

    Returns
    -------
    np.ndarray
        Shape (time_frames, n_mels + n_prosodic), dtype float32.

    Example
    -------
    >>> spec = np.zeros((94, 128))   # 3s audio at 16kHz, hop=512
    >>> pvec = np.zeros((12,))
    >>> combined = append_prosody_to_spectrogram(spec, pvec)
    >>> combined.shape
    (94, 140)
    """
    if spectrogram.ndim != 2:
        raise ValueError(
            f"[normalization] spectrogram must be 2D, got shape {spectrogram.shape}"
        )
    if prosodic_vec.ndim != 1:
        raise ValueError(
            f"[normalization] prosodic_vec must be 1D, got shape {prosodic_vec.shape}"
        )

    time_frames = spectrogram.shape[0]
    # Broadcast prosodic vector across time axis
    prosody_tiled = np.tile(prosodic_vec, (time_frames, 1))  # (T, n_prosodic)
    combined = np.concatenate([spectrogram, prosody_tiled], axis=1)
    return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 50)
    print("  ProsodicNormalizer self-test")
    print("=" * 50)

    rng = np.random.default_rng(42)

    # Simulate Tamil calibration data: 5 samples, 12 features
    # (small — triggers warning)
    X_calib_small = rng.normal(loc=150.0, scale=20.0, size=(5, 12)).astype(np.float32)

    print("\n--- Test 1: small sample warning ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        norm = ProsodicNormalizer(language="tamil_demo", method="zscore")
        norm.fit(X_calib_small)
        assert len(w) >= 1, "Expected a small-sample warning"
        print(f"[OK] Warning raised: {str(w[0].message)[:60]}...")

    # Test transform
    X_test = rng.normal(loc=150.0, scale=20.0, size=(3, 12)).astype(np.float32)
    X_norm = norm.transform(X_test)
    assert X_norm.shape == X_test.shape
    assert X_norm.dtype == np.float32
    assert np.all(np.isfinite(X_norm))
    print(f"[OK] transform: shape={X_norm.shape}")

    # Test single-sample transform
    X_single = X_test[0]
    X_single_norm = norm.transform(X_single)
    assert X_single_norm.shape == (12,)
    print(f"[OK] single-sample transform: shape={X_single_norm.shape}")

    # Test save/load roundtrip
    print("\n--- Test 2: save/load roundtrip ---")
    X_calib_large = rng.normal(loc=150.0, scale=20.0, size=(50, 12)).astype(np.float32)
    norm_large = ProsodicNormalizer(language="english", method="zscore")
    norm_large.fit(X_calib_large)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_norm.json"
        norm_large.save(save_path)
        norm_loaded = ProsodicNormalizer.load(save_path)

        X_via_orig = norm_large.transform(X_test)
        X_via_load = norm_loaded.transform(X_test)
        assert np.allclose(X_via_orig, X_via_load, atol=1e-5), "Save/load mismatch"
        print("[OK] save/load roundtrip: values match")

    # Test fit_transform
    X_ft = norm_large.fit_transform(X_calib_large)
    assert X_ft.shape == X_calib_large.shape
    print(f"[OK] fit_transform: shape={X_ft.shape}")

    # Test append_prosody_to_spectrogram
    spec = np.zeros((94, 128), dtype=np.float32)
    pvec = np.ones(12, dtype=np.float32)
    combined = append_prosody_to_spectrogram(spec, pvec)
    assert combined.shape == (94, 140), f"Wrong shape: {combined.shape}"
    print(f"[OK] append_prosody_to_spectrogram: shape={combined.shape}")

    # Summary
    print("\n" + norm_large.summary())
    print("\n[PASS] normalization.py self-test complete.")
