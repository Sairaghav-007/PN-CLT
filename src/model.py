"""
src/model.py
============
CNN-BiLSTM architecture for speech emotion recognition.

Architecture Summary
--------------------

  Input: (batch, TIME_STEPS, FEATURES)
         TIME_STEPS = 94  (3s audio @ 16kHz, hop=512)
         FEATURES   = 140 (128 Mel + 12 prosodic)

  ┌─────────────────────────────────────────────────────┐
  │ BLOCK 1 — Local acoustic pattern detection          │
  │   Conv1D(128, kernel=5, padding='same')             │
  │   BatchNormalization()                              │
  │   Activation('relu')                               │
  │   MaxPooling1D(pool_size=2) → TIME_STEPS/2         │
  │   Dropout(0.3)                                      │
  ├─────────────────────────────────────────────────────┤
  │ BLOCK 2 — Higher-level spectral patterns            │
  │   Conv1D(256, kernel=3, padding='same')             │
  │   BatchNormalization()                              │
  │   Activation('relu')                               │
  │   MaxPooling1D(pool_size=2) → TIME_STEPS/4         │
  │   Dropout(0.3)                                      │
  ├─────────────────────────────────────────────────────┤
  │ TEMPORAL MODELLING                                  │
  │   Bidirectional(LSTM(128, return_sequences=False))  │
  │   → captures forward/backward temporal context     │
  ├─────────────────────────────────────────────────────┤
  │ CLASSIFICATION HEAD                                 │
  │   Dense(128, relu)                                 │
  │   Dropout(0.4)                                      │
  │   Dense(64, relu)                                  │
  │   Dropout(0.3)                                      │
  │   Dense(N_CLASSES, softmax)  ← swappable for TL    │
  └─────────────────────────────────────────────────────┘

Transfer Learning Layer Groups
-------------------------------
  frozen_core   : [conv1d_block1, bn1, conv1d_block2, bn2]
  trainable_top : [bidirectional, dense_1, dense_2, output]

Public API
----------
build_cnn_bilstm(input_shape, n_classes, cfg_model) -> keras.Model
replace_classification_head(model, n_classes, cfg_model) -> keras.Model
get_transfer_layer_groups(model) -> dict[str, list[str]]
freeze_layers(model, layer_names) -> None
unfreeze_layers(model, layer_names) -> None
compile_model(model, lr, n_classes) -> None
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Layer name constants — used by transfer learning code
# ---------------------------------------------------------------------------

# Layers frozen in Phase 1 of fine-tuning
FROZEN_CORE_LAYERS = [
    "cnn_block1_conv",
    "cnn_block1_bn",
    "cnn_block2_conv",
    "cnn_block2_bn",
]

# Layers progressively unfrozen in Phase 2 of fine-tuning
UPPER_LAYERS = [
    "bilstm",
    "dense_1",
    "dropout_dense1",
    "dense_2",
    "dropout_dense2",
    "output_emotion",
]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_cnn_bilstm(
    input_shape: tuple[int, int],
    n_classes: int,
    # Architecture hyperparameters (can be overridden from config)
    cnn_filters_1: int = 128,
    cnn_filters_2: int = 256,
    cnn_kernel_1: int = 5,
    cnn_kernel_2: int = 3,
    pool_size: int = 2,
    dropout_cnn: float = 0.3,
    lstm_units: int = 128,
    dense_units_1: int = 128,
    dense_units_2: int = 64,
    dropout_dense_1: float = 0.4,
    dropout_dense_2: float = 0.3,
    name: str = "cnn_bilstm_ser",
) -> keras.Model:
    """
    Build the CNN-BiLSTM speech emotion recognition model.

    Parameters
    ----------
    input_shape : tuple (time_steps, features)
        Shape of one input sample (excluding batch dimension).
        e.g., (94, 140) for 3s audio with 128 Mel + 12 prosodic.
    n_classes : int
        Number of emotion classes (output units).
    cnn_filters_1, cnn_filters_2 : int
        Number of filters in Conv1D blocks 1 and 2.
    cnn_kernel_1, cnn_kernel_2 : int
        Kernel sizes for Conv1D blocks 1 and 2.
    pool_size : int
        MaxPooling window size (applied after each CNN block).
    dropout_cnn : float
        Dropout rate after each CNN block.
    lstm_units : int
        Number of units in each direction of the BiLSTM.
        (Total hidden state = lstm_units * 2)
    dense_units_1, dense_units_2 : int
        Units in the two dense layers.
    dropout_dense_1, dropout_dense_2 : float
        Dropout rates after dense layers.
    name : str
        Model name (shown in .summary()).

    Returns
    -------
    keras.Model
        Uncompiled model. Call compile_model() to add optimizer/loss.

    Notes
    -----
    - Layer names are hardcoded and referenced by transfer learning code.
      Do NOT rename them without updating FROZEN_CORE_LAYERS and UPPER_LAYERS.
    - The model uses the Functional API (not Sequential) for flexibility
      in layer freezing and head replacement.
    """
    inputs = keras.Input(shape=input_shape, name="input_features")

    # ---------------------------------------------------------------
    # BLOCK 1 — Local acoustic / spectral pattern detection
    # Captures short-duration phoneme-level features (5 frames ≈ 160ms)
    # ---------------------------------------------------------------
    x = layers.Conv1D(
        filters=cnn_filters_1,
        kernel_size=cnn_kernel_1,
        padding="same",
        name="cnn_block1_conv",
    )(inputs)
    x = layers.BatchNormalization(name="cnn_block1_bn")(x)
    x = layers.Activation("relu", name="cnn_block1_relu")(x)
    x = layers.MaxPooling1D(pool_size=pool_size, name="cnn_block1_pool")(x)
    x = layers.Dropout(dropout_cnn, name="cnn_block1_drop")(x)

    # ---------------------------------------------------------------
    # BLOCK 2 — Higher-level spectral/rhythmic patterns
    # Captures word-level or phrase-level features
    # ---------------------------------------------------------------
    x = layers.Conv1D(
        filters=cnn_filters_2,
        kernel_size=cnn_kernel_2,
        padding="same",
        name="cnn_block2_conv",
    )(x)
    x = layers.BatchNormalization(name="cnn_block2_bn")(x)
    x = layers.Activation("relu", name="cnn_block2_relu")(x)
    x = layers.MaxPooling1D(pool_size=pool_size, name="cnn_block2_pool")(x)
    x = layers.Dropout(dropout_cnn, name="cnn_block2_drop")(x)

    # ---------------------------------------------------------------
    # TEMPORAL MODELLING — BiLSTM
    # Learns temporal evolution of emotional speech (forward + backward)
    # BiLSTM output = 2 * lstm_units (concatenation of both directions)
    # ---------------------------------------------------------------
    x = layers.Bidirectional(
        layers.LSTM(units=lstm_units, return_sequences=False),
        name="bilstm",
    )(x)

    # ---------------------------------------------------------------
    # CLASSIFICATION HEAD
    # Dense layers + Dropout for final emotion discrimination
    # ---------------------------------------------------------------
    x = layers.Dense(dense_units_1, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_dense_1, name="dropout_dense1")(x)
    x = layers.Dense(dense_units_2, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_dense_2, name="dropout_dense2")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output_emotion")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ---------------------------------------------------------------------------
# Head replacement (for transfer learning)
# ---------------------------------------------------------------------------

def replace_classification_head(
    model: keras.Model,
    n_classes: int,
    dense_units_1: int = 128,
    dense_units_2: int = 64,
    dropout_dense_1: float = 0.4,
    dropout_dense_2: float = 0.3,
) -> keras.Model:
    """
    Remove the existing classification head and attach a new one.

    Used in transfer learning when the target language (Tamil) has a
    different number of emotion classes than the source (English), or
    simply to reinitialize the head with random weights before fine-tuning.

    The backbone (CNN + BiLSTM) weights are PRESERVED.

    Parameters
    ----------
    model : keras.Model
        The pretrained model (e.g., English-trained CNN-BiLSTM).
    n_classes : int
        Number of output emotion classes for the new head.
    dense_units_1, dense_units_2 : int
        New dense layer sizes (may differ from original).
    dropout_dense_1, dropout_dense_2 : float
        New dropout rates.

    Returns
    -------
    keras.Model
        New model with the same backbone + fresh classification head.
    """
    # Find the BiLSTM output — this is where we cut and reattach
    try:
        bilstm_layer = model.get_layer("bilstm")
    except ValueError as exc:
        raise ValueError(
            "[model] Could not find 'bilstm' layer in the pretrained model. "
            "Make sure the model was built with build_cnn_bilstm()."
        ) from exc

    # Build new head on top of the BiLSTM output
    backbone_output = bilstm_layer.output

    x = layers.Dense(dense_units_1, activation="relu", name="dense_1")(backbone_output)
    x = layers.Dropout(dropout_dense_1, name="dropout_dense1")(x)
    x = layers.Dense(dense_units_2, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_dense_2, name="dropout_dense2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output_emotion")(x)

    new_model = keras.Model(
        inputs=model.inputs,
        outputs=outputs,
        name=model.name + "_adapted",
    )
    return new_model


# ---------------------------------------------------------------------------
# Transfer learning helpers
# ---------------------------------------------------------------------------

def get_transfer_layer_groups(model: keras.Model) -> dict[str, list[str]]:
    """
    Return a dictionary mapping transfer-learning stage names to layer name lists.

    Returns
    -------
    dict with keys:
        "frozen_core"   — always frozen in fine-tuning (lower CNN blocks)
        "upper_layers"  — unfrozen in phase-2 fine-tuning
        "all_names"     — all layer names (for reference)
    """
    return {
        "frozen_core": FROZEN_CORE_LAYERS,
        "upper_layers": UPPER_LAYERS,
        "all_names": [layer.name for layer in model.layers],
    }


def freeze_layers(model: keras.Model, layer_names: list[str]) -> None:
    """
    Freeze specified layers (set trainable=False).

    Freezing stops gradient updates for these layers during fine-tuning,
    preserving the English acoustic representations learned during pretraining.

    Parameters
    ----------
    model : keras.Model
    layer_names : list of str
        Exact layer names to freeze. Partial matches are skipped with a warning.
    """
    frozen_count = 0
    for name in layer_names:
        try:
            layer = model.get_layer(name)
            layer.trainable = False
            frozen_count += 1
        except ValueError:
            # Layer may not exist if head was replaced
            pass
    # Also freeze any layers whose name STARTS WITH any of the names
    # (handles BatchNorm momentum layers etc.)
    for layer in model.layers:
        for target_name in layer_names:
            if layer.name.startswith(target_name) and layer.trainable:
                layer.trainable = False
                frozen_count += 1
    print(f"[model] Froze {frozen_count} layer(s): {layer_names}")


def unfreeze_layers(model: keras.Model, layer_names: list[str]) -> None:
    """
    Unfreeze specified layers (set trainable=True) for phase-2 fine-tuning.

    Parameters
    ----------
    model : keras.Model
    layer_names : list of str
        Exact layer names to unfreeze.
    """
    unfrozen_count = 0
    for name in layer_names:
        try:
            layer = model.get_layer(name)
            layer.trainable = True
            unfrozen_count += 1
        except ValueError:
            pass
    print(f"[model] Unfroze {unfrozen_count} layer(s): {layer_names}")


def print_trainability(model: keras.Model) -> None:
    """
    Print trainable status of each layer.
    Useful for verifying freeze/unfreeze is applied correctly.
    """
    print(f"\n{'Layer Name':<30} {'Trainable':>10} {'Parameters':>12}")
    print("-" * 54)
    for layer in model.layers:
        params = layer.count_params()
        status = "YES" if layer.trainable else "FROZEN"
        print(f"  {layer.name:<28} {status:>10} {params:>12,}")
    total = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(p) for p in model.trainable_weights
    )
    print("-" * 54)
    print(f"  {'TOTAL':<28} {'':>10} {total:>12,}")
    print(f"  {'TRAINABLE':<28} {'':>10} {trainable:>12,}")
    print(f"  {'FROZEN':<28} {'':>10} {total - trainable:>12,}\n")


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_model(
    model: keras.Model,
    lr: float = 0.001,
    n_classes: int = 7,
    label_smoothing: float = 0.0,
) -> None:
    """
    Compile the model with Adam optimizer and categorical cross-entropy.

    Uses sparse_categorical_crossentropy so labels can be plain integers
    (no need to one-hot encode).

    Parameters
    ----------
    model : keras.Model
    lr : float
        Learning rate for Adam optimizer.
    n_classes : int
        Number of classes (used to select appropriate loss).
    label_smoothing : float
        Label smoothing factor (0 = no smoothing).
        Useful for regularization in small datasets.
    """
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    if label_smoothing > 0.0:
        # CategoricalCrossentropy supports label smoothing,
        # but requires one-hot labels. Wrap in a lambda loss instead.
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        print(
            f"[model] Note: label_smoothing={label_smoothing} ignored with "
            "SparseCategoricalCrossentropy. Use one-hot encoding to enable it."
        )
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def get_callbacks(
    checkpoint_path: str,
    patience_stop: int = 15,
    patience_lr: int = 7,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    verbose: int = 1,
) -> list:
    """
    Return a standard set of Keras callbacks for training.

    Includes:
    - ModelCheckpoint: saves best model by val_loss
    - EarlyStopping: stops if val_loss doesn't improve
    - ReduceLROnPlateau: halves LR when val_loss plateaus
    - TensorBoard: optional training visualization

    Parameters
    ----------
    checkpoint_path : str
        Where to save the best model checkpoint (.keras format).
    patience_stop : int
        EarlyStopping patience (epochs).
    patience_lr : int
        ReduceLROnPlateau patience (epochs).
    lr_factor : float
        Factor by which LR is reduced.
    min_lr : float
        Minimum learning rate.
    verbose : int
        Verbosity level (0=silent, 1=one line per epoch).

    Returns
    -------
    list of keras.callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose,
            mode="min",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_stop,
            restore_best_weights=True,
            verbose=verbose,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=verbose,
            mode="min",
        ),
    ]
    return callbacks


# ---------------------------------------------------------------------------
# Model from config
# ---------------------------------------------------------------------------

def build_model_from_config(cfg) -> keras.Model:
    """
    Build a CNN-BiLSTM model using all parameters from config.yaml.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object from src/config.py.

    Returns
    -------
    keras.Model (uncompiled)
    """
    input_shape = cfg.get_input_shape()
    n_classes = cfg.get_n_classes()
    m = cfg.model

    return build_cnn_bilstm(
        input_shape=input_shape,
        n_classes=n_classes,
        cnn_filters_1=m.cnn_filters_1,
        cnn_filters_2=m.cnn_filters_2,
        cnn_kernel_1=m.cnn_kernel_1,
        cnn_kernel_2=m.cnn_kernel_2,
        pool_size=m.pool_size,
        dropout_cnn=m.dropout_cnn,
        lstm_units=m.lstm_units,
        dense_units_1=m.dense_units_1,
        dense_units_2=m.dense_units_2,
        dropout_dense_1=m.dropout_dense_1,
        dropout_dense_2=m.dropout_dense_2,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import numpy as np

    print("=" * 60)
    print("  Model self-test")
    print("=" * 60)

    INPUT_SHAPE = (94, 140)   # 3s @ 16kHz hop=512, 128 Mel + 12 prosodic
    N_CLASSES = 7
    BATCH = 4

    # --- Build model ---
    model = build_cnn_bilstm(input_shape=INPUT_SHAPE, n_classes=N_CLASSES)
    model.summary(line_length=70)

    # --- Compile ---
    compile_model(model, lr=0.001, n_classes=N_CLASSES)
    print("[OK] Model compiled")

    # --- Forward pass ---
    dummy_X = np.random.randn(BATCH, *INPUT_SHAPE).astype(np.float32)
    dummy_y = np.array([0, 1, 2, 3], dtype=np.int32)
    preds = model.predict(dummy_X, verbose=0)
    assert preds.shape == (BATCH, N_CLASSES), f"Wrong output shape: {preds.shape}"
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"
    print(f"[OK] Forward pass: output shape={preds.shape}")
    print(f"     Probabilities sum: {preds.sum(axis=1)}")

    # --- Loss step ---
    loss_val = model.evaluate(dummy_X, dummy_y, verbose=0)
    print(f"[OK] evaluate: loss={loss_val[0]:.4f}, acc={loss_val[1]:.4f}")

    # --- Layer groups ---
    groups = get_transfer_layer_groups(model)
    print(f"[OK] Layer groups: frozen_core={groups['frozen_core']}")

    # --- Freeze/unfreeze ---
    freeze_layers(model, FROZEN_CORE_LAYERS)
    compile_model(model, lr=1e-4)
    print_trainability(model)
    unfreeze_layers(model, UPPER_LAYERS)
    compile_model(model, lr=5e-5)
    print("[OK] freeze/unfreeze cycle")

    # --- Head replacement ---
    model_5class = replace_classification_head(model, n_classes=5)
    compile_model(model_5class, lr=1e-4, n_classes=5)
    preds_5 = model_5class.predict(dummy_X, verbose=0)
    assert preds_5.shape == (BATCH, 5), f"Head replace shape: {preds_5.shape}"
    print(f"[OK] replace_classification_head: output shape={preds_5.shape}")

    # --- Callbacks ---
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cbs = get_callbacks(f"{tmp}/best.keras", patience_stop=5)
        assert len(cbs) == 3
        print(f"[OK] get_callbacks: {len(cbs)} callbacks")

    print("\n[PASS] model.py self-test complete.")
