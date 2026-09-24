from pathlib import Path
import json
import csv

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "realsign_enhanced"
OUTPUT_DIR = BASE_DIR / "ml" / "trained"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
VAL_CSV = DATA_DIR / "validation.csv"
LABELS_PATH = DATA_DIR / "labels.json"

MODEL_PATH = OUTPUT_DIR / "signsync_realsign_enhanced.keras"
LABELS_OUTPUT = OUTPUT_DIR / "labels_realsign_enhanced.json"
NORMALIZATION_OUTPUT = OUTPUT_DIR / "realsign_enhanced_normalization.npz"

BATCH_SIZE = 64
EPOCHS = 50
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    X = []
    y = []

    with open(path, "r", encoding="utf-8") as f:

        reader = csv.reader(f)
        header = next(reader)

        for row in reader:

            y.append(row[0])

            X.append([
                float(value)
                for value in row[1:]
            ])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y)
    )


print()
print("=" * 70)
print("SIGN SYNC - ENHANCED REALSIGN MODEL TRAINING")
print("=" * 70)
print()


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:

    labels = json.load(f)

label_to_index = {
    label: index
    for index, label in enumerate(labels)
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

X_train, y_train_text = load_csv(TRAIN_CSV)

print("Loading testing data...")

X_test, y_test_text = load_csv(TEST_CSV)

print("Loading validation data...")

X_val, y_val_text = load_csv(VAL_CSV)


y_train = np.array([
    label_to_index[label]
    for label in y_train_text
])

y_test = np.array([
    label_to_index[label]
    for label in y_test_text
])

y_val = np.array([
    label_to_index[label]
    for label in y_val_text
])


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Validation samples:", len(X_val))
print("Features:", X_train.shape[1])
print("Classes:", len(labels))
print("Labels:", labels)


# ============================================================
# NORMALIZATION
# ============================================================

print()
print("Calculating normalization...")

feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)

feature_std[feature_std < 1e-6] = 1.0

X_train = (
    X_train - feature_mean
) / feature_std

X_test = (
    X_test - feature_mean
) / feature_std

X_val = (
    X_val - feature_mean
) / feature_std


# ============================================================
# MODEL
# ============================================================

print()
print("Building model...")

model = keras.Sequential([

    layers.Input(shape=(126,)),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.20),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.15),

    layers.Dense(
        len(labels),
        activation="softmax"
    )
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print()
print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Test accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASS-BY-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("CLASS-BY-CLASS ACCURACY")
print("=" * 70)

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

for index, label in enumerate(labels):

    mask = (
        y_test == index
    )

    if np.sum(mask) == 0:
        continue

    accuracy = np.mean(
        predicted_classes[mask] == index
    )

    print(
        f"{label}: {accuracy * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# SAVE LABELS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    LABELS_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        labels,
        f,
        indent=2
    )


# ============================================================
# SAVE NORMALIZATION
# ============================================================

np.savez(
    NORMALIZATION_OUTPUT,
    mean=feature_mean.astype(np.float32),
    std=feature_std.astype(np.float32)
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Labels:")
print(LABELS_OUTPUT)

print()
print("Normalization:")
print(NORMALIZATION_OUTPUT)

print()
print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("IMPORTANT:")
print("The existing live model was NOT replaced.")
print()