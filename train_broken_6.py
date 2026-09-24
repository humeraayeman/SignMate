from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "realsign"
OUT_DIR = BASE_DIR / "ml" / "trained"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ONLY these letters are being repaired.
TARGETS = ["I", "O", "Q", "R", "T", "W"]

# Everything else becomes OTHER.
SPECIAL_CLASSES = [
    "OTHER",
    "I",
    "O",
    "Q",
    "R",
    "T",
    "W"
]

print("=" * 60)
print("SIGN SYNC - BROKEN 6 LETTER REPAIR")
print("=" * 60)
print("Repairing ONLY:", TARGETS)
print()


def load_csv(name):
    path = DATA_DIR / name
    print("Loading:", path)
    return pd.read_csv(path)


train_df = load_csv("train.csv")
test_df = load_csv("test.csv")
val_df = load_csv("validation.csv")


# ------------------------------------------------------------
# Find feature columns
# ------------------------------------------------------------

feature_columns = [
    c for c in train_df.columns
    if c.startswith("f")
]

print("Features:", len(feature_columns))


def convert_labels(df):

    df = df.copy()

    df["repair_label"] = df["label"].apply(
        lambda x: x if x in TARGETS else "OTHER"
    )

    return df


train_df = convert_labels(train_df)
test_df = convert_labels(test_df)
val_df = convert_labels(val_df)


# ------------------------------------------------------------
# Prepare X/y
# ------------------------------------------------------------

X_train = train_df[feature_columns].values.astype(
    np.float32
)

X_test = test_df[feature_columns].values.astype(
    np.float32
)

X_val = val_df[feature_columns].values.astype(
    np.float32
)


label_to_index = {
    label: i
    for i, label in enumerate(SPECIAL_CLASSES)
}

y_train = train_df["repair_label"].map(
    label_to_index
).values.astype(np.int32)

y_test = test_df["repair_label"].map(
    label_to_index
).values.astype(np.int32)

y_val = val_df["repair_label"].map(
    label_to_index
).values.astype(np.int32)


# ------------------------------------------------------------
# Standardization
# ------------------------------------------------------------

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std < 1e-6] = 1.0

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_val = (X_val - mean) / std


# ------------------------------------------------------------
# Print class counts
# ------------------------------------------------------------

print()
print("TRAINING CLASS COUNTS")

for label in SPECIAL_CLASSES:
    count = int(
        np.sum(
            y_train == label_to_index[label]
        )
    )
    print(f"{label:>6}: {count}")

print()


# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------

classes_present = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_present,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(
        classes_present,
        weights
    )
}

# Give the six actual repair letters a little extra weight.
for letter in TARGETS:
    idx = label_to_index[letter]

    if idx in class_weights:
        class_weights[idx] *= 1.5


print("CLASS WEIGHTS:")
print(class_weights)
print()


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(len(feature_columns),)
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
        len(SPECIAL_CLASSES),
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


model.summary()


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

model_path = (
    OUT_DIR
    / "signsync_broken_6.keras"
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),

    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

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

    callbacks=callbacks,

    verbose=1
)


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Overall test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# PER-CLASS VALIDATION ACCURACY
# ------------------------------------------------------------

predictions = model.predict(
    X_val,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

print()
print("VALIDATION ACCURACY BY CLASS")

for label in SPECIAL_CLASSES:

    idx = label_to_index[label]

    mask = y_val == idx

    if np.sum(mask) == 0:
        continue

    accuracy = np.mean(
        predicted_classes[mask] == idx
    )

    print(
        f"{label:>6}: "
        f"{accuracy * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ------------------------------------------------------------
# SAVE LABELS
# ------------------------------------------------------------

labels_path = (
    OUT_DIR
    / "broken_6_labels.json"
)

with open(
    labels_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        SPECIAL_CLASSES,
        f,
        indent=2
    )


# ------------------------------------------------------------
# SAVE NORMALIZATION
# ------------------------------------------------------------

normalization_path = (
    OUT_DIR
    / "broken_6_normalization.npz"
)

np.savez(
    normalization_path,
    mean=mean.astype(np.float32),
    std=std.astype(np.float32)
)


print()
print("=" * 60)
print("BROKEN 6 MODEL COMPLETE")
print("=" * 60)

print("Model:")
print(model_path)

print("Labels:")
print(labels_path)

print("Normalization:")
print(normalization_path)

print()
print("ONLY repaired:")
print(", ".join(TARGETS))

print()
print("Original A-Z model was NOT modified.")
print("=" * 60)