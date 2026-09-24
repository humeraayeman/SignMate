from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ml" / "data" / "prathum_ck"
OUT_DIR = BASE_DIR / "ml" / "trained"

OUT_DIR.mkdir(parents=True, exist_ok=True)

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

print("Labels:", labels)
print("Train:", X_train.shape)
print("Test:", X_test.shape)
print("Validation:", X_val.shape)

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

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(126,)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(
        len(labels),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
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
    )
]

print("\nTraining C/K specialist model...\n")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nC/K TEST ACCURACY: {accuracy * 100:.2f}%")

model_path = OUT_DIR / "signsync_prathum_ck.keras"
labels_path = OUT_DIR / "prathum_ck_labels.json"

model.save(model_path)

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(labels, f)

print("\nSaved model:")
print(model_path)

print("\nSaved labels:")
print(labels_path)

print("\nC/K TRAINING COMPLETE")