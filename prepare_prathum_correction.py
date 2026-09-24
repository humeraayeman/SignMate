from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "ml"
    / "data"
    / "prathum_correction"
)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
VAL_PATH = DATA_DIR / "validation.csv"

LABELS_PATH = DATA_DIR / "labels.json"

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "trained"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "signsync_prathum_correction.keras"
)

OUTPUT_LABELS_PATH = (
    MODEL_DIR
    / "prathum_correction_labels.json"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:

    labels = json.load(f)


print()
print("=" * 60)
print("SIGNMATE - TRAINING CORRECTION MODEL")
print("=" * 60)

print("Classes:")
print(labels)

print("Number of classes:", len(labels))


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)

val_df = pd.read_csv(
    VAL_PATH
)


# ============================================================
# SPLIT FEATURES / LABELS
# ============================================================

feature_columns = [
    f"f{i}"
    for i in range(126)
]


X_train = train_df[
    feature_columns
].values.astype(
    np.float32
)

X_test = test_df[
    feature_columns
].values.astype(
    np.float32
)

X_val = val_df[
    feature_columns
].values.astype(
    np.float32
)


# ============================================================
# ENCODE LABELS
# ============================================================

label_to_index = {
    label: index
    for index, label in enumerate(labels)
}


y_train = np.array([
    label_to_index[label]
    for label in train_df["label"]
])

y_test = np.array([
    label_to_index[label]
    for label in test_df["label"]
])

y_val = np.array([
    label_to_index[label]
    for label in val_df["label"]
])


# ============================================================
# DISPLAY DATASET SIZE
# ============================================================

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Validation samples:", len(X_val))

print(
    "Input features:",
    X_train.shape[1]
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(labels)),
    y=y_train
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}

print()
print("Class weights:")

for index, label in enumerate(labels):

    print(
        f"{label}: "
        f"{class_weights[index]:.3f}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(126,)
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.25
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.15
    ),

    tf.keras.layers.Dense(
        len(labels),
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=8,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath=str(
        MODEL_PATH
    ),

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("TESTING")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print()
print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)


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

    total = int(
        np.sum(mask)
    )

    if total == 0:
        continue

    correct = int(
        np.sum(
            predicted_classes[mask]
            == index
        )
    )

    accuracy = (
        correct / total
    ) * 100

    print(
        f"{label}: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# SAVE LABELS
# ============================================================

with open(
    OUTPUT_LABELS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        labels,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Labels:",
    OUTPUT_LABELS_PATH
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("=" * 60)