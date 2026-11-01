"""
Phase 2 smoke test — run this directly to verify environment:
    python phase2_smoke_test.py
Expected output: all [OK] lines.
"""

import sys
print(f"Python: {sys.version}")

# ---- Standard library ----
import os, pathlib, json, math, random, warnings
print("[OK] Standard library")

# ---- NumPy ----
import numpy as np
print(f"[OK] numpy {np.__version__}")

# ---- SciPy ----
import scipy
print(f"[OK] scipy {scipy.__version__}")

# ---- LibROSA ----
import librosa
print(f"[OK] librosa {librosa.__version__}")

# ---- SoundFile ----
import soundfile as sf
print(f"[OK] soundfile {sf.__version__}")

# ---- scikit-learn ----
import sklearn
print(f"[OK] scikit-learn {sklearn.__version__}")

# ---- pandas ----
import pandas as pd
print(f"[OK] pandas {pd.__version__}")

# ---- matplotlib ----
import matplotlib
matplotlib.use("Agg")   # non-interactive for smoke test
import matplotlib.pyplot as plt
print(f"[OK] matplotlib {matplotlib.__version__}")

# ---- seaborn ----
import seaborn as sns
print(f"[OK] seaborn {sns.__version__}")

# ---- PyYAML ----
import yaml
print(f"[OK] PyYAML")

# ---- joblib ----
import joblib
print(f"[OK] joblib {joblib.__version__}")

# ---- TensorFlow ----
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    # tf.version.VERSION works for both tensorflow and tensorflow-cpu on Windows
    tf_version = getattr(tf, '__version__', None) or tf.version.VERSION
    print(f"[OK] TensorFlow {tf_version}")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"     GPUs available: {len(gpus)}")
    except Exception:
        print("     GPU detection skipped.")
except ImportError as e:
    print(f"[WARN] TensorFlow not installed: {e}")

# ---- Parselmouth (optional) ----
try:
    import parselmouth    print(f"[OK] parselmouth (Praat) {parselmouth.__version__}")
except ImportError:
    print("[WARN] parselmouth not available — will use librosa for pitch extraction")

# ---- Streamlit ----
try:
    import streamlit
    print(f"[OK] streamlit {streamlit.__version__}")
except ImportError as e:
    print(f"[WARN] streamlit not installed: {e}")

# ---- click ----
try:
    import click
    print(f"[OK] click {click.__version__}")
except ImportError as e:
    print(f"[WARN] click not installed: {e}")

# ---- Config module ----
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.config import cfg
print("[OK] Config loaded")
print(cfg.summary())

# ---- Verify input shape calculation ----
shape = cfg.get_input_shape()
assert len(shape) == 2, "Input shape must be 2D"
print(f"[OK] Expected input shape: {shape}")

print("\n=== Phase 2 Smoke Test PASSED ===")
