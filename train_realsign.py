from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "ml"
    / "data"
    / "realsign"
)

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
VAL_FILE = DATA_DIR / "validation.csv"

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "trained"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "signsync_realsign.keras"
)

LABELS_FILE = (
    MODEL_DIR
    / "labels_realsign.json"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64

EPOCHS = 30

RANDOM_SEED = 42


np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("SIGN SYNC - REALSIGN CLASSIFIER TRAINING")
print("=" * 60)
print()

print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

print(
    "Training rows:",
    len(train_df)
)

print("Loading testing data...")

test_df = pd.read_csv(
    TEST_FILE
)

print(
    "Testing rows:",
    len(test_df)
)

print("Loading validation data...")

val_df = pd.read_csv(
    VAL_FILE
)

print(
    "Validation rows:",
    len(val_df)
)


# ============================================================
# FEATURES / LABELS
# ============================================================

feature_columns = [
    column
    for column in train_df.columns
    if column.startswith("feature_")
]

print()
print(
    "Number of features:",
    len(feature_columns)
)


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


y_train_text = train_df[
    "label"
].values

y_test_text = test_df[
    "label"
].values

y_val_text = val_df[
    "label"
].values


# ============================================================
# LABEL ENCODING
# ============================================================

labels = sorted(
    np.unique(y_train_text).tolist()
)

print()
print(
    "Classes:",
    labels
)

print(
    "Number of classes:",
    len(labels)
)


label_to_index = {
    label: index
    for index, label in enumerate(labels)
}


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


# ============================================================
# NORMALIZATION
# ============================================================

print()
print("Calculating feature normalization...")

feature_mean = X_train.mean(
    axis=0
)

feature_std = X_train.std(
    axis=0
)

feature_std[
    feature_std < 1e-6
] = 1.0


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
print("Building neural network...")


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
        0.25
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.20
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


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
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

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_FILE),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
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

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TEST")
print("=" * 60)
print()

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    predicted_classes
)

print()
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y_test,
        predicted_classes,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0
    )
)


# ============================================================
# SAVE LABELS
# ============================================================

with open(
    LABELS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        labels,
        f,
        indent=2
    )


# ============================================================
# SAVE NORMALIZATION VALUES
# ============================================================

NORMALIZATION_FILE = (
    MODEL_DIR
    / "realsign_normalization.npz"
)

np.savez(
    NORMALIZATION_FILE,
    mean=feature_mean,
    std=feature_std
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print()

print(
    "Model:"
)

print(
    MODEL_FILE
)

print()

print(
    "Labels:"
)

print(
    LABELS_FILE
)

print()

print(
    "Normalization:"
)

print(
    NORMALIZATION_FILE
)

print()
print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("=" * 60)