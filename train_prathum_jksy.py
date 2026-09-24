from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "jksy"
MODEL_DIR = BASE_DIR / "ml" / "trained"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

train = np.load(DATA_DIR / "train.npz")
test = np.load(DATA_DIR / "test.npz")
validation = np.load(DATA_DIR / "validation.npz")

X_train = train["X"]
y_train = train["y"]

X_test = test["X"]
y_test = test["y"]

X_val = validation["X"]
y_val = validation["y"]


with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)


print("=" * 60)
print("J K S Y SPECIALIST MODEL")
print("=" * 60)

print("Labels:", labels)
print("Train:", X_train.shape)
print("Test:", X_test.shape)
print("Validation:", X_val.shape)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print()
print("Class weights:", class_weights)


# ============================================================
# MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(126,)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        4,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR /
    "signsync_prathum_jksy.keras"
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("TRAINING...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=80,
    batch_size=32,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TEST")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print()
print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE LABELS
# ============================================================

labels_path = (
    MODEL_DIR /
    "prathum_jksy_labels.json"
)

with open(
    labels_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        labels,
        f,
        indent=2
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    best_model_path
)


print()
print("=" * 60)
print("J K S Y TRAINING COMPLETE")
print("=" * 60)
print("Model:", best_model_path)
print("Labels:", labels_path)
print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)
print("=" * 60)