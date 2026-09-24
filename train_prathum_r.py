from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "prathum_targets"
OUT_DIR = BASE_DIR / "ml" / "trained"

FEATURES = [f"f{i}" for i in range(126)]

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")


# ============================================================
# Keep all six classes, but heavily oversample R
# ============================================================

r_train = train_df[train_df["label"] == "R"]

other_train = train_df[train_df["label"] != "R"]

# Repeat R several times.
r_extra = pd.concat(
    [r_train] * 4,
    ignore_index=True
)

train_aug = pd.concat(
    [other_train, r_extra],
    ignore_index=True
)

train_aug = train_aug.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


labels = ["I", "O", "R", "T", "V", "Y"]

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}


def prepare(df):

    X = df[FEATURES].values.astype(np.float32)

    y = np.array([
        label_to_id[x]
        for x in df["label"]
    ])

    return X, y


X_train, y_train = prepare(train_aug)
X_val, y_val = prepare(val_df)
X_test, y_test = prepare(test_df)


print("=" * 60)
print("R-FOCUSED TRAINING")
print("=" * 60)

print("Original R:", len(r_train))
print("R after oversampling:", len(r_extra))
print("Total training:", len(X_train))
print()


# ============================================================
# Model
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(126,)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        6,
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


model_path = (
    OUT_DIR /
    "signsync_prathum_targets_r.keras"
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        mode="max"
    ),

    tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]


model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# Evaluation
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("=" * 60)
print(f"Overall test accuracy: {accuracy * 100:.2f}%")
print("=" * 60)


# ============================================================
# Per-class accuracy
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

predicted = np.argmax(
    predictions,
    axis=1
)

for label in labels:

    class_id = label_to_id[label]

    mask = y_test == class_id

    if np.sum(mask) == 0:
        continue

    class_accuracy = np.mean(
        predicted[mask] == class_id
    )

    print(
        f"{label}: "
        f"{class_accuracy * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# Labels
# ============================================================

with open(
    OUT_DIR / "prathum_targets_r_labels.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(labels, f, indent=2)


print()
print("Model saved:")
print(model_path)