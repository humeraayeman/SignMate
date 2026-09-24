from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "realsign"
OUTPUT_DIR = BASE_DIR / "ml" / "trained"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"

MODEL_FILE = OUTPUT_DIR / "signsync_target_5.keras"
LABELS_FILE = OUTPUT_DIR / "target_5_labels.json"
NORMALIZATION_FILE = OUTPUT_DIR / "target_5_normalization.npz"


TARGETS = ["O", "Q", "R", "V", "Y"]

CLASSES = [
    "OTHER",
    "O",
    "Q",
    "R",
    "V",
    "Y"
]


print("=" * 70)
print("SIGN SYNC - O Q R V Y TARGET CORRECTION")
print("=" * 70)


train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)


# ------------------------------------------------------------
# Find columns
# ------------------------------------------------------------

label_column = "label"

if label_column not in train_df.columns:
    for name in ["class", "target"]:
        if name in train_df.columns:
            label_column = name
            break

feature_columns = []

for column in train_df.columns:
    if column == label_column:
        continue

    if pd.api.types.is_numeric_dtype(train_df[column]):
        feature_columns.append(column)


print("Features:", len(feature_columns))


# ------------------------------------------------------------
# Convert labels
# ------------------------------------------------------------

def convert_label(value):
    value = str(value).strip().upper()

    if value in TARGETS:
        return value

    return "OTHER"


train_labels = train_df[label_column].apply(convert_label)
val_labels = val_df[label_column].apply(convert_label)


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

X_train_df = train_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X_val_df = val_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

train_mask = ~X_train_df.isna().any(axis=1)
val_mask = ~X_val_df.isna().any(axis=1)

X_train = X_train_df.loc[train_mask].to_numpy(
    dtype=np.float32
)

X_val = X_val_df.loc[val_mask].to_numpy(
    dtype=np.float32
)

train_labels = train_labels.loc[
    train_mask
].reset_index(drop=True)

val_labels = val_labels.loc[
    val_mask
].reset_index(drop=True)


label_to_index = {
    label: i
    for i, label in enumerate(CLASSES)
}

y_train = np.array(
    [
        label_to_index[x]
        for x in train_labels
    ],
    dtype=np.int32
)

y_val = np.array(
    [
        label_to_index[x]
        for x in val_labels
    ],
    dtype=np.int32
)


# ------------------------------------------------------------
# Show target counts
# ------------------------------------------------------------

print()
print("Training samples:")

for label in CLASSES:
    count = int(
        np.sum(train_labels == label)
    )

    print(
        f"  {label:>5}: {count}"
    )

print()


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

mean = np.mean(
    X_train,
    axis=0
).astype(np.float32)

std = np.std(
    X_train,
    axis=0
).astype(np.float32)

std[std < 1e-6] = 1.0

X_train = (
    X_train - mean
) / std

X_val = (
    X_val - mean
) / std


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.savez(
    NORMALIZATION_FILE,
    mean=mean,
    std=std
)


# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------

present = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=present,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(
        present,
        weights
    )
}


# Slightly emphasize ONLY the five target signs.
for label in TARGETS:

    index = label_to_index[label]

    if index in class_weights:
        class_weights[index] *= 1.75


print("Class weights:")

for i, label in enumerate(CLASSES):

    print(
        f"  {label:>5}: "
        f"{class_weights.get(i, 1.0):.3f}"
    )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(
        len(CLASSES),
        activation="softmax"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILE,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=60,

    batch_size=32,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_FILE
)

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Validation accuracy: "
    f"{accuracy * 100:.2f}%"
)


predictions = model.predict(
    X_val,
    verbose=0
)

predicted = np.argmax(
    predictions,
    axis=1
)


print()
print("CLASS ACCURACY")
print("-" * 40)

for index, label in enumerate(CLASSES):

    mask = y_val == index

    if np.sum(mask) == 0:
        continue

    acc = np.mean(
        predicted[mask] == index
    )

    print(
        f"{label:>5}: "
        f"{acc * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ------------------------------------------------------------
# Save labels
# ------------------------------------------------------------

with open(
    LABELS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        CLASSES,
        f,
        indent=4
    )


print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print("Model:")
print(MODEL_FILE)

print()
print("Labels:")
print(LABELS_FILE)

print()
print("Normalization:")
print(NORMALIZATION_FILE)

print()
print("TARGETS ONLY:")
print("O, Q, R, V, Y")
print()
print("Original RealSign model was NOT modified.")