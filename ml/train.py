import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# SIGN-SYNC NORMALIZED MODEL TRAINING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "ml"
    / "data"
    / "landmarks_normalized.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "trained"
)

MODEL_FILE = (
    MODEL_DIR
    / "signsync_alphabet_normalized.keras"
)

LABEL_FILE = (
    MODEL_DIR
    / "labels_normalized.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("==============================================")
print(" SignSync - Normalized A-Z Model Training")
print("==============================================")
print()

if not DATA_FILE.exists():

    print("ERROR: Dataset file not found:")
    print(DATA_FILE)
    exit()


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna()

if len(df) == 0:

    print("ERROR: No valid data.")
    exit()


# ============================================================
# FEATURES
# ============================================================

X = (
    df
    .drop(columns=["label"])
    .values
    .astype(np.float32)
)


# ============================================================
# LABELS
# ============================================================

y_text = (
    df["label"]
    .astype(str)
    .values
)


label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)

labels = label_encoder.classes_.tolist()


print()
print("Classes:")

for index, label in enumerate(labels):

    print(
        f"{index}: {label}"
    )


print()
print("Number of classes:", len(labels))
print("Number of features:", X.shape[1])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(126,)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.35
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(labels),
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


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Starting training...")
print()

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# EVALUATION
# ============================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print()
print("==============================================")
print(" Model evaluation")
print("==============================================")
print()

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Test loss: {loss:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_FILE
)


with open(
    LABEL_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        labels,
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================================")
print(" Normalized training completed")
print("==============================================")
print()

print("Model:")
print(MODEL_FILE)

print()

print("Labels:")
print(LABEL_FILE)

print()

print(
    f"Final test accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()