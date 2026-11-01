"""
src/dataset.py
==============
Dataset builders for English (TESS/SAVEE) and Tamil audio files.

Responsibilities:
  1. Scan dataset directories and collect (file_path, emotion_label) pairs
  2. Apply audio preprocessing (load, resample, pad/truncate, normalize)
  3. Extract Log-Mel spectral features
  4. Extract prosodic features
  5. Append prosodic vector to spectrogram frames
  6. Encode emotion labels to integers
  7. Return train/val/test numpy arrays ready for the model

IMPORTANT DATA-LEAKAGE RULES enforced here:
  - ProsodicNormalizer is NEVER fitted on val/test data
  - Spectrogram statistics (if global norm) computed on train only
  - Label encoder fitted on train labels only (Tamil uses English encoder)

Public API
----------
TESSDataset(tess_dir, cfg)           — scan + load TESS
SAVEEDataset(savee_dir, cfg)         — scan + load SAVEE (optional)
TamilDataset(tamil_dir, cfg)         — scan + load Tamil files
build_english_dataset(cfg)           — returns X_train, X_val, X_test, y_*
detect_tamil_mode(tamil_dir)         — returns ('demo'|'poc'|'few_shot'|'full', N)
build_tamil_dataset(cfg, normalizer) — returns X_all, y_all (with normalizer applied)
save_label_encoder(le, path)
load_label_encoder(path)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import cfg as _cfg
from src.feature_extraction import extract_features, normalize_spectrogram
from src.normalization import ProsodicNormalizer, append_prosody_to_spectrogram
from src.preprocessing import batch_preprocess, load_audio, split_indices
from src.prosody import extract_prosodic_batch, extract_prosodic_features


# ---------------------------------------------------------------------------
# Tamil mode detection thresholds
# ---------------------------------------------------------------------------
DEMO_THRESHOLD = 1
POC_THRESHOLD = 19
FEW_SHOT_THRESHOLD = 99


# ---------------------------------------------------------------------------
# Tamil mode detection
# ---------------------------------------------------------------------------

def detect_tamil_mode(tamil_dir: str | Path) -> tuple[str, int]:
    """
    Detect operation mode based on how many Tamil WAV files are present.

    Returns
    -------
    (mode, n_files) where mode is one of:
        'demo'      — 1 file: inference only, no fine-tuning
        'poc'       — 2–19 files: proof-of-concept, no valid split
        'few_shot'  — 20–99 files: few-shot experiment
        'full'      — 100+ files: full experiment with few-shot curves
    """
    tamil_dir = Path(tamil_dir)
    if not tamil_dir.exists():
        return ("demo", 0)

    wav_files = sorted(list(set(tamil_dir.rglob("*.wav")) | set(tamil_dir.rglob("*.WAV"))))
    n = len(wav_files)

    if n <= DEMO_THRESHOLD:
        mode = "demo"
    elif n <= POC_THRESHOLD:
        mode = "poc"
    elif n <= FEW_SHOT_THRESHOLD:
        mode = "few_shot"
    else:
        mode = "full"

    _print_mode_banner(mode, n)
    return (mode, n)


def _print_mode_banner(mode: str, n: int) -> None:
    border = "=" * 65
    print(f"\n{border}")
    print(f"  TAMIL DATASET MODE DETECTION")
    print(f"  Found {n} Tamil WAV file(s)")
    print(f"  Mode: {mode.upper()}")
    if mode == "demo":
        print("  ACTION: Inference/demo only. No fine-tuning possible.")
        print("  NOTE:   1 sample cannot produce reliable statistics.")
    elif mode == "poc":
        print("  ACTION: Proof-of-concept adaptation (all files used).")
        print("  WARNING: Results are NOT statistically meaningful.")
        print("           Do not report accuracy as validated performance.")
    elif mode == "few_shot":
        print("  ACTION: Few-shot experiment. Train/val/test split applied.")
        print("  NOTE:   Results still limited by small dataset size.")
    else:
        print("  ACTION: Full experiment with few-shot learning curves.")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# TESS dataset scanner
# ---------------------------------------------------------------------------

def scan_tess(tess_dir: str | Path, label_map: dict[str, str]) -> list[tuple[Path, str]]:
    """
    Scan the TESS dataset directory and return (file_path, emotion) pairs.

    TESS folder structure:
        TESS/OAF_angry/OAF_angry_01.wav
        TESS/YAF_happy/YAF_happy_01.wav
        ...

    Emotion is extracted from the folder name (e.g. 'OAF_angry' -> 'angry').
    'ps' (pleasant surprise) is remapped to 'surprise'.

    Parameters
    ----------
    tess_dir : str or Path
        Root directory of the TESS dataset.
    label_map : dict
        Mapping from TESS emotion name to canonical label.

    Returns
    -------
    list of (Path, str) tuples.
    """
    tess_dir = Path(tess_dir)
    if not tess_dir.exists():
        raise FileNotFoundError(
            f"[dataset] TESS directory not found: {tess_dir}\n"
            "Please download TESS from:\n"
            "  https://tspace.library.utoronto.ca/handle/1807/24487\n"
            "and place it in: data/english/TESS/"
        )

    samples: list[tuple[Path, str]] = []
    unknown_emotions: set[str] = set()

    all_wavs = sorted(
        list(set(tess_dir.rglob("*.wav")) | set(tess_dir.rglob("*.WAV")))
    )
    known_keys = sorted(label_map.keys(), key=len, reverse=True)

    for wav_file in all_wavs:
        canonical = None
        # 1. Try matching emotion from parent directory name (e.g. OAF_angry or angry)
        parent_name = wav_file.parent.name.lower()
        parent_parts = parent_name.split("_")
        raw_emotion = parent_parts[-1] if len(parent_parts) >= 2 else parent_name

        if raw_emotion in label_map:
            canonical = label_map[raw_emotion]
        else:
            for k in known_keys:
                if k in parent_name:
                    canonical = label_map[k]
                    break

        # 2. Try matching emotion from filename (e.g. OAF_back_angry.wav)
        if canonical is None:
            stem = wav_file.stem.lower()
            stem_parts = stem.split("_")
            raw_emotion_file = stem_parts[-1] if len(stem_parts) >= 2 else stem
            if raw_emotion_file in label_map:
                canonical = label_map[raw_emotion_file]
            else:
                for k in known_keys:
                    if k in stem:
                        canonical = label_map[k]
                        break

        if canonical is not None:
            samples.append((wav_file, canonical))
        else:
            unknown_emotions.add(raw_emotion)

    if unknown_emotions:
        warnings.warn(
            f"[dataset] TESS: Unrecognised emotion folders: {unknown_emotions}. "
            "Add mappings to config.yaml → emotions.tamil_label_map",
            stacklevel=2,
        )

    print(f"[dataset] TESS: Found {len(samples)} audio files across "
          f"{len(set(e for _, e in samples))} emotion classes.")
    return samples


# ---------------------------------------------------------------------------
# SAVEE dataset scanner (optional)
# ---------------------------------------------------------------------------

def scan_savee(savee_dir: str | Path) -> list[tuple[Path, str]]:
    """
    Scan the SAVEE dataset directory.

    SAVEE structure: SAVEE/<speaker>/<emotion_code><n>.wav
    Emotion codes: a=anger, d=disgust, f=fear, h=happiness,
                   n=neutral, sa=sadness, su=surprise

    Parameters
    ----------
    savee_dir : str or Path
        Root SAVEE directory.

    Returns
    -------
    list of (Path, str) tuples.
    """
    savee_dir = Path(savee_dir)
    if not savee_dir.exists():
        warnings.warn(
            f"[dataset] SAVEE directory not found: {savee_dir}. "
            "Skipping SAVEE. Only TESS will be used for English training.",
            stacklevel=2,
        )
        return []

    code_map = {
        "a": "angry", "d": "disgust", "f": "fear",
        "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise",
    }

    samples: list[tuple[Path, str]] = []
    for speaker_dir in sorted(savee_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for wav_file in sorted(speaker_dir.glob("*.wav")):
            stem = wav_file.stem.lower()
            # Match longest emotion code prefix (handles 'sa' before 'a')
            matched = None
            for code in sorted(code_map.keys(), key=len, reverse=True):
                if stem.startswith(code):
                    matched = code_map[code]
                    break
            if matched:
                samples.append((wav_file, matched))

    print(f"[dataset] SAVEE: Found {len(samples)} audio files.")
    return samples


# ---------------------------------------------------------------------------
# Tamil dataset scanner
# ---------------------------------------------------------------------------

def scan_tamil(
    tamil_dir: str | Path,
    label_map: dict[str, str],
) -> list[tuple[Path, str]]:
    """
    Scan the Tamil dataset directory.

    Supported naming conventions (auto-detected):
      1. Filename contains emotion: tamil_happy_01.wav  → 'happy'
      2. Parent folder is emotion: happy/file.wav       → 'happy'
      3. No label found: returns (path, 'unknown') with warning

    Parameters
    ----------
    tamil_dir : str or Path
    label_map : dict
        Canonical emotion name mapping.

    Returns
    -------
    list of (Path, str) tuples. Files without labels get 'unknown'.
    """
    tamil_dir = Path(tamil_dir)
    if not tamil_dir.exists():
        warnings.warn(
            f"[dataset] Tamil directory not found: {tamil_dir}. "
            "Please place Tamil WAV files in data/tamil/",
            stacklevel=2,
        )
        return []

    wav_files = sorted(
        list(set(tamil_dir.rglob("*.wav")) | set(tamil_dir.rglob("*.WAV")))
    )

    if not wav_files:
        warnings.warn(
            f"[dataset] No WAV files found in: {tamil_dir}",
            stacklevel=2,
        )
        return []

    canonical_labels = set(label_map.values())
    samples: list[tuple[Path, str]] = []
    no_label_count = 0

    for wav_file in wav_files:
        emotion = _infer_label_from_path(wav_file, label_map, canonical_labels)
        if emotion == "unknown":
            no_label_count += 1
        samples.append((wav_file, emotion))

    if no_label_count > 0:
        warnings.warn(
            f"[dataset] Tamil: {no_label_count} file(s) could not be labelled. "
            "They will be assigned 'unknown' and excluded from training.\n"
            "Name files as: <emotion>_<speaker>_<n>.wav (e.g. happy_speaker1_01.wav)\n"
            "or place them in emotion-named subdirectories.",
            stacklevel=2,
        )

    labelled = [(p, e) for p, e in samples if e != "unknown"]
    print(f"[dataset] Tamil: {len(wav_files)} WAV files found, "
          f"{len(labelled)} labelled, {no_label_count} unlabelled.")
    return samples


def _infer_label_from_path(
    path: Path,
    label_map: dict[str, str],
    canonical_labels: set[str],
) -> str:
    """Try to infer emotion label from file path or parent directory name."""
    name_lower = path.stem.lower()
    parent_lower = path.parent.name.lower()

    # Check canonical labels and mapped aliases in filename
    all_known = set(label_map.keys()) | canonical_labels
    for candidate in sorted(all_known, key=len, reverse=True):
        if candidate in name_lower:
            return label_map.get(candidate, candidate)

    # Check parent directory
    for candidate in sorted(all_known, key=len, reverse=True):
        if candidate in parent_lower:
            return label_map.get(candidate, candidate)

    return "unknown"


# ---------------------------------------------------------------------------
# Feature extraction pipeline (shared by English and Tamil)
# ---------------------------------------------------------------------------

def extract_all_features(
    file_paths: list[Path],
    sr: int,
    duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    f0_min: float,
    f0_max: float,
    frame_length: float,
    hop_length_s: float,
    normalize_spec: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full feature extraction pipeline for a list of audio files.

    Steps:
      1. Load and preprocess each audio file
      2. Extract Log-Mel spectrogram (MFSC)
      3. Optionally normalize spectrogram per-file
      4. Extract prosodic feature vector
      5. Return spectrograms and prosodic vectors separately

    Parameters
    ----------
    file_paths : list of Path
    sr, duration, n_mels, n_fft, hop_length, fmin, fmax : audio/feature params
    f0_min, f0_max, frame_length, hop_length_s : prosodic params
    normalize_spec : bool
        If True, apply per-file z-score normalization to spectrogram.
    verbose : bool

    Returns
    -------
    spectrograms : np.ndarray, shape (N, time_frames, n_mels)
    prosodic_vecs : np.ndarray, shape (N, n_prosodic)
    """
    target_len = int(sr * duration)

    all_specs = []
    all_pros = []
    skipped = 0

    for i, fpath in enumerate(file_paths):
        if verbose and i % 50 == 0:
            print(f"  [dataset] Feature extraction: {i}/{len(file_paths)}", flush=True)
        try:
            audio = load_audio(fpath, sr=sr, duration=duration, normalize=True)
            # Spectral
            spec = extract_features(
                audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
                hop_length=hop_length, fmin=fmin, fmax=fmax,
            )
            if normalize_spec:
                spec = normalize_spectrogram(spec)
            # Prosodic
            pvec = extract_prosodic_features(
                audio, sr=sr, f0_min=f0_min, f0_max=f0_max,
                frame_length=frame_length, hop_length_s=hop_length_s,
            )
            all_specs.append(spec)
            all_pros.append(pvec)
        except Exception as exc:
            warnings.warn(f"[dataset] Skipping {fpath.name}: {exc}", stacklevel=2)
            skipped += 1

    if skipped > 0:
        print(f"[dataset] Skipped {skipped} files due to errors.")

    if len(all_specs) == 0:
        return np.zeros((0, 94, n_mels), dtype=np.float32), np.zeros((0, 12), dtype=np.float32)

    spectrograms = np.stack(all_specs, axis=0)      # (N, T, n_mels)
    prosodic_vecs = np.stack(all_pros, axis=0)       # (N, n_prosodic)
    return spectrograms, prosodic_vecs


def combine_features(
    spectrograms: np.ndarray,
    prosodic_vecs: np.ndarray,
    normalizer: Optional[ProsodicNormalizer] = None,
) -> np.ndarray:
    """
    Combine spectrograms and prosodic vectors into model-ready input.

    Optionally applies ProsodicNormalizer to prosodic vectors before
    appending them to spectrogram frames.

    Parameters
    ----------
    spectrograms : np.ndarray, shape (N, T, n_mels)
    prosodic_vecs : np.ndarray, shape (N, n_prosodic)
    normalizer : ProsodicNormalizer or None
        If provided, normalize prosodic vectors first.

    Returns
    -------
    np.ndarray, shape (N, T, n_mels + n_prosodic)
    """
    if len(spectrograms) == 0:
        n_pros = prosodic_vecs.shape[1] if (hasattr(prosodic_vecs, "ndim") and prosodic_vecs.ndim > 1) else 12
        n_mels = spectrograms.shape[2] if (hasattr(spectrograms, "ndim") and spectrograms.ndim > 2) else 128
        return np.zeros((0, 94, n_mels + n_pros), dtype=np.float32)

    if normalizer is not None:
        prosodic_vecs = normalizer.transform(prosodic_vecs)

    combined = []
    for i in range(len(spectrograms)):
        c = append_prosody_to_spectrogram(spectrograms[i], prosodic_vecs[i])
        combined.append(c)
    return np.stack(combined, axis=0)  # (N, T, n_mels + n_prosodic)


# ---------------------------------------------------------------------------
# English dataset builder
# ---------------------------------------------------------------------------

def build_english_dataset(
    cfg=None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           LabelEncoder]:
    """
    Build the complete English training dataset from TESS (+ optional SAVEE).

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, T, F)
    y_train, y_val, y_test : np.ndarray of int
    label_encoder : LabelEncoder (fitted on training labels)

    Notes
    -----
    - ProsodicNormalizer for English is fitted on X_train prosodic only.
    - English normalizer is saved to disk for reference but is NOT used
      in the Tamil pipeline (Tamil gets its own normalizer).
    - The label encoder is saved to disk immediately after fitting.
    """
    if cfg is None:
        cfg = _cfg

    cfg.ensure_dirs()

    # --- Scan datasets ---
    label_map = cfg.get_tess_label_map()
    tess_dir = Path(cfg.paths.english_tess)
    savee_dir = Path(cfg.paths.english_savee)

    samples = scan_tess(tess_dir, label_map)
    samples += scan_savee(savee_dir)   # empty list if SAVEE not present

    if not samples:
        raise RuntimeError(
            "[dataset] No English audio files found. "
            "Please download TESS and place it in:\n"
            f"  {tess_dir}"
        )

    file_paths = [p for p, _ in samples]
    raw_labels = [e for _, e in samples]

    # --- Label encoding ---
    le = LabelEncoder()
    # Fit only on the known canonical classes from config, not just what's present
    # This ensures consistent integer↔label mapping across runs
    le.fit(sorted(cfg.emotions.classes))
    y_encoded = le.transform(raw_labels)

    print(f"[dataset] English: {len(samples)} samples across "
          f"{len(le.classes_)} classes: {list(le.classes_)}")

    # --- Train/val/test split (index-based, reproducible) ---
    train_idx, val_idx, test_idx = split_indices(
        len(samples),
        val_frac=cfg.training.val_split,
        test_frac=cfg.training.test_split,
        seed=cfg.seed,
    )

    train_paths = [file_paths[i] for i in train_idx]
    val_paths   = [file_paths[i] for i in val_idx]
    test_paths  = [file_paths[i] for i in test_idx]

    y_train = y_encoded[train_idx]
    y_val   = y_encoded[val_idx]
    y_test  = y_encoded[test_idx]

    # --- Feature extraction ---
    a = cfg.audio
    f = cfg.features
    p = cfg.prosody

    def _extract(paths, tag):
        if verbose:
            print(f"[dataset] Extracting features: {tag} ({len(paths)} files)...")
        specs, pros = extract_all_features(
            paths, sr=a.sample_rate, duration=a.duration,
            n_mels=f.n_mels, n_fft=f.n_fft, hop_length=f.hop_length,
            fmin=f.fmin, fmax=f.fmax,
            f0_min=p.f0_min, f0_max=p.f0_max,
            frame_length=p.frame_length, hop_length_s=p.hop_length_s,
            verbose=verbose,
        )
        return specs, pros

    specs_train, pros_train = _extract(train_paths, "TRAIN")
    specs_val,   pros_val   = _extract(val_paths,   "VAL")
    specs_test,  pros_test  = _extract(test_paths,  "TEST")

    # --- English ProsodicNormalizer (fitted on TRAIN only) ---
    eng_normalizer = ProsodicNormalizer(language="english", method="zscore")
    eng_normalizer.fit(pros_train, feature_names=["pitch_mean","pitch_std",
        "pitch_range","voiced_fraction","energy_mean","energy_std","energy_max",
        "zcr_mean","zcr_std","silence_ratio","pitch_slope","energy_slope"])
    eng_normalizer.save(cfg.paths.normalizer_english)

    # --- Combine spectrograms + normalised prosodic ---
    X_train = combine_features(specs_train, pros_train, eng_normalizer)
    X_val   = combine_features(specs_val,   pros_val,   eng_normalizer)
    X_test  = combine_features(specs_test,  pros_test,  eng_normalizer)

    print(f"\n[dataset] English dataset shapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # --- Save label encoder ---
    save_label_encoder(le, cfg.paths.label_encoder)

    # --- Save preprocessing config (for inference consistency) ---
    _save_preprocessing_config(cfg)

    return X_train, X_val, X_test, y_train, y_val, y_test, le


# ---------------------------------------------------------------------------
# Tamil dataset builder
# ---------------------------------------------------------------------------

def build_tamil_dataset(
    cfg=None,
    normalizer: Optional[ProsodicNormalizer] = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[int], list[str], str]:
    """
    Build Tamil dataset with mode-aware handling.

    Parameters
    ----------
    cfg : Config
    normalizer : ProsodicNormalizer or None
        If None, prosodic features are used without normalization (Model A/B).
        If provided (Model C/D), Tamil prosodic normalization is applied.

    Returns
    -------
    X_all : list of np.ndarray
        Each element shape (T, F) — combined spectrogram + prosodic.
    y_all : list of int
        Encoded emotion labels (-1 for unlabelled files).
    file_names : list of str
    mode : str
        One of 'demo', 'poc', 'few_shot', 'full'.
    """
    if cfg is None:
        cfg = _cfg

    tamil_dir = Path(cfg.paths.tamil_data)
    label_map_raw = cfg.emotions.tamil_label_map
    label_map = {k: v for k, v in vars(label_map_raw).items()}

    mode, n_files = detect_tamil_mode(tamil_dir)

    if n_files == 0:
        print("[dataset] No Tamil files found. Returning empty dataset.")
        return [], [], [], mode

    samples = scan_tamil(tamil_dir, label_map)

    # Load label encoder (must exist — trained on English)
    try:
        le = load_label_encoder(cfg.paths.label_encoder)
    except FileNotFoundError:
        # Fallback: create fresh encoder with config classes
        warnings.warn(
            "[dataset] Label encoder not found. Creating fresh encoder from config.",
            stacklevel=2,
        )
        le = LabelEncoder()
        le.fit(sorted(cfg.emotions.classes))

    a = cfg.audio
    f = cfg.features
    p = cfg.prosody

    X_all = []
    y_all = []
    file_names = []

    for fpath, emotion in samples:
        try:
            audio = load_audio(fpath, sr=a.sample_rate,
                               duration=a.duration, normalize=True)
            spec = extract_features(
                audio, sr=a.sample_rate, n_mels=f.n_mels,
                n_fft=f.n_fft, hop_length=f.hop_length,
                fmin=f.fmin, fmax=f.fmax,
            )
            spec = normalize_spectrogram(spec)
            pvec = extract_prosodic_features(
                audio, sr=a.sample_rate, f0_min=p.f0_min, f0_max=p.f0_max,
                frame_length=p.frame_length, hop_length_s=p.hop_length_s,
            )

            # Apply Tamil prosodic normalization if normalizer is provided
            if normalizer is not None:
                pvec_norm = normalizer.transform(pvec.reshape(1, -1))[0]
            else:
                pvec_norm = pvec

            combined = append_prosody_to_spectrogram(spec, pvec_norm)
            X_all.append(combined)

            # Encode label
            if emotion == "unknown":
                y_all.append(-1)
            elif emotion in le.classes_:
                y_all.append(int(le.transform([emotion])[0]))
            else:
                warnings.warn(
                    f"[dataset] Tamil emotion '{emotion}' not in label encoder. "
                    f"Known: {list(le.classes_)}. Assigning -1.",
                    stacklevel=2,
                )
                y_all.append(-1)

            file_names.append(fpath.name)

        except Exception as exc:
            warnings.warn(f"[dataset] Skipping Tamil file {fpath.name}: {exc}",
                          stacklevel=2)

    print(f"[dataset] Tamil dataset built: {len(X_all)} samples loaded.")
    return X_all, y_all, file_names, mode


# ---------------------------------------------------------------------------
# Few-shot sampling
# ---------------------------------------------------------------------------

def sample_few_shot(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    n_classes: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample exactly n_per_class examples per class.

    Parameters
    ----------
    X : np.ndarray, shape (N, ...)
    y : np.ndarray, shape (N,) of int
    n_per_class : int
    n_classes : int
    seed : int

    Returns
    -------
    X_few, y_few : np.ndarray (may have fewer than n_per_class * n_classes
                               if some classes have insufficient samples)
    """
    rng = np.random.default_rng(seed)
    selected_idx = []
    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            warnings.warn(f"[dataset] Class {cls} has no samples.", stacklevel=2)
            continue
        n_take = min(n_per_class, len(cls_idx))
        if n_take < n_per_class:
            warnings.warn(
                f"[dataset] Class {cls}: requested {n_per_class} samples but "
                f"only {len(cls_idx)} available. Taking all {n_take}.",
                stacklevel=2,
            )
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        selected_idx.extend(chosen.tolist())

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)
    return X[selected_idx], y[selected_idx]


# ---------------------------------------------------------------------------
# Label encoder I/O
# ---------------------------------------------------------------------------

def save_label_encoder(le: LabelEncoder, path: str | Path) -> None:
    """Save LabelEncoder to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, path)
    print(f"[dataset] Label encoder saved: {path}")
    print(f"         Classes: {list(le.classes_)}")


def load_label_encoder(path: str | Path) -> LabelEncoder:
    """Load LabelEncoder from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"[dataset] Label encoder not found: {path}\n"
            "Run train_english.py first to create it."
        )
    le = joblib.load(path)
    print(f"[dataset] Label encoder loaded: {list(le.classes_)}")
    return le


# ---------------------------------------------------------------------------
# Preprocessing config save/load
# ---------------------------------------------------------------------------

def _save_preprocessing_config(cfg) -> None:
    """Save preprocessing parameters to JSON for inference consistency."""
    config_data = {
        "sample_rate": cfg.audio.sample_rate,
        "duration": cfg.audio.duration,
        "n_mels": cfg.features.n_mels,
        "n_fft": cfg.features.n_fft,
        "hop_length": cfg.features.hop_length,
        "fmin": cfg.features.fmin,
        "fmax": cfg.features.fmax,
        "f0_min": cfg.prosody.f0_min,
        "f0_max": cfg.prosody.f0_max,
        "n_prosodic_features": cfg.prosody.n_prosodic_features,
        "n_classes": cfg.emotions.n_classes,
        "emotion_classes": cfg.emotions.classes,
    }
    path = Path(cfg.paths.preprocessing_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"[dataset] Preprocessing config saved: {path}")
