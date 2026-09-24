from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SIGN SYNC - TARGETED CORRECTION MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "realsign"
OUTPUT_DIR = BASE_DIR / "ml" / "trained"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"

MODEL_FILE = OUTPUT_DIR / "signsync_target_correction.keras"
LABELS_FILE = OUTPUT_DIR / "target_correction_labels.json"
NORMALIZATION_FILE = OUTPUT_DIR / "target_correction_normalization.npz"


# Only these signs are being targeted.
TARGET_LABELS = [
    "I",
    "L",
    "O",
    "Q",
    "S",
    "T",
    "V",
    "Y",
]

CORRECTION_LABELS = [
    "OTHER",
    "I",
    "L",
    "O",
    "Q",
    "S",
    "T",
    "V",
    "Y",
]


print()
print("=" * 70)
print("SIGN SYNC - TARGETED CORRECTION MODEL")
print("=" * 70)
print()

# ============================================================
# CHECK FILES
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not VAL_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VAL_FILE}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Training file:")
print(TRAIN_FILE)
print()

print("Validation file:")
print(VAL_FILE)
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_FILE)

print("Loading validation data...")

val_df = pd.read_csv(VAL_FILE)

print()
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print()


# ============================================================
# FIND FEATURE COLUMNS
# ============================================================

# The prepared RealSign CSV contains 126 landmark features.
# We detect them automatically instead of depending on exact
# column names.

NON_FEATURE_COLUMNS = {
    "label",
    "class",
    "target",
    "image",
    "image_path",
    "filename",
    "file",
    "path",
}


def find_label_column(df):
    possible = [
        "label",
        "class",
        "target",
    ]

    for column in possible:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find label column in CSV.\n"
        f"Available columns:\n{list(df.columns)}"
    )


train_label_column = find_label_column(train_df)
val_label_column = find_label_column(val_df)


feature_columns = []

for column in train_df.columns:
    if column in NON_FEATURE_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(train_df[column]):
        feature_columns.append(column)


if len(feature_columns) != 126:
    print(
        f"WARNING: Expected 126 numeric feature columns, "
        f"but found {len(feature_columns)}."
    )

print("Label column:", train_label_column)
print("Feature columns:", len(feature_columns))
print()


if len(feature_columns) == 0:
    raise ValueError(
        "No numeric feature columns were found."
    )


# ============================================================
# PREPARE LABELS
# ============================================================

def convert_to_correction_label(label):
    label = str(label).strip().upper()

    if label in TARGET_LABELS:
        return label

    return "OTHER"


train_labels = train_df[train_label_column].apply(
    convert_to_correction_label
)

val_labels = val_df[val_label_column].apply(
    convert_to_correction_label
)


print("Target correction classes:")
for label in CORRECTION_LABELS:
    count = int((train_labels == label).sum())
    print(f"  {label:>5}: {count}")

print()


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

train_features_df = train_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

val_features_df = val_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

train_valid_mask = ~train_features_df.isna().any(axis=1)
val_valid_mask = ~val_features_df.isna().any(axis=1)

train_features_df = train_features_df.loc[
    train_valid_mask
].reset_index(drop=True)

val_features_df = val_features_df.loc[
    val_valid_mask
].reset_index(drop=True)

train_labels = train_labels.loc[
    train_valid_mask
].reset_index(drop=True)

val_labels = val_labels.loc[
    val_valid_mask
].reset_index(drop=True)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train = train_features_df.to_numpy(
    dtype=np.float32
)

X_val = val_features_df.to_numpy(
    dtype=np.float32
)


label_to_index = {
    label: index
    for index, label in enumerate(CORRECTION_LABELS)
}

y_train = np.array(
    [
        label_to_index[label]
        for label in train_labels
    ],
    dtype=np.int32
)

y_val = np.array(
    [
        label_to_index[label]
        for label in val_labels
    ],
    dtype=np.int32
)


print("Prepared training shape:", X_train.shape)
print("Prepared validation shape:", X_val.shape)
print()


# ============================================================
# STANDARDIZATION
# ============================================================

print("Calculating normalization statistics...")

feature_mean = np.mean(
    X_train,
    axis=0
).astype(np.float32)

feature_std = np.std(
    X_train,
    axis=0
).astype(np.float32)

feature_std[
    feature_std < 1e-6
] = 1.0


X_train = (
    X_train - feature_mean
) / feature_std

X_val = (
    X_val - feature_mean
) / feature_std


X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)


# ============================================================
# SAVE NORMALIZATION
# ============================================================

np.savez(
    NORMALIZATION_FILE,
    mean=feature_mean,
    std=feature_std
)

print(
    "Normalization saved:",
    NORMALIZATION_FILE
)
print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

all_classes = np.arange(
    len(CORRECTION_LABELS)
)

present_classes = np.unique(
    y_train
)

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=present_classes,
    y=y_train
)

class_weights = {}

for class_index, weight in zip(
    present_classes,
    balanced_weights
):
    class_weights[int(class_index)] = float(weight)


# Give the targeted signs slightly more importance.
for target_label in TARGET_LABELS:

    target_index = label_to_index[target_label]

    if target_index in class_weights:
        class_weights[target_index] *= 1.5


print()
print("Class weights:")

for index, label in enumerate(CORRECTION_LABELS):

    weight = class_weights.get(
        index,
        1.0
    )

    print(
        f"  {label:>5}: {weight:.4f}"
    )

print()


# ============================================================
# BUILD MODEL
# ============================================================

print("=" * 70)
print("BUILDING TARGET CORRECTION MODEL")
print("=" * 70)
print()

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(X_train.shape[1],)
        ),

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.15
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.10
        ),

        tf.keras.layers.Dense(
            len(CORRECTION_LABELS),
            activation="softmax"
        ),
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_FILE,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING TARGET CORRECTION TRAINING")
print("=" * 70)
print()

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=60,

    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)
print()

loss, accuracy = best_model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(
    f"Validation loss: {loss:.4f}"
)

print(
    f"Validation accuracy: {accuracy * 100:.2f}%"
)

print()


# ============================================================
# SAVE LABELS
# ============================================================

with open(
    LABELS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        CORRECTION_LABELS,
        f,
        indent=4
    )


print(
    "Labels saved:",
    LABELS_FILE
)


# ============================================================
# CLASS-BY-CLASS VALIDATION
# ============================================================

print()
print("=" * 70)
print("CLASS-BY-CLASS ACCURACY")
print("=" * 70)
print()

predictions = best_model.predict(
    X_val,
    verbose=0
)

predicted_indices = np.argmax(
    predictions,
    axis=1
)


for index, label in enumerate(
    CORRECTION_LABELS
):

    mask = y_val == index

    if np.sum(mask) == 0:
        print(
            f"{label:>5}: no validation samples"
        )
        continue

    class_accuracy = np.mean(
        predicted_indices[mask] == index
    )

    print(
        f"{label:>5}: "
        f"{class_accuracy * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TARGET CORRECTION TRAINING COMPLETE")
print("=" * 70)
print()

print("Created files:")

print(
    "  Model:",
    MODEL_FILE
)

print(
    "  Labels:",
    LABELS_FILE
)

print(
    "  Normalization:",
    NORMALIZATION_FILE
)

print()
print("Target signs:")
print(
    "  I, L, O, Q, S, T, V, Y"
)

print()
print("The original RealSign model was NOT modified.")
print()
print("=" * 70)