import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "prathum_correction"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
VAL_CSV = DATA_DIR / "validation.csv"
LABELS_JSON = DATA_DIR / "labels.json"

OUTPUT_DIR = BASE_DIR / "ml" / "trained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "signsync_prathum_correction.keras"
LABELS_OUT = OUTPUT_DIR / "prathum_correction_labels.json"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PRATHUM ISL CORRECTION MODEL TRAINING")
print("=" * 60)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
val_df = pd.read_csv(VAL_CSV)

with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels_data = json.load(f)

print(f"Training samples:   {len(train_df)}")
print(f"Testing samples:    {len(test_df)}")
print(f"Validation samples: {len(val_df)}")

# labels.json may contain either a list or a dictionary
if isinstance(labels_data, list):
    labels = labels_data
else:
    labels = labels_data.get("labels", labels_data)

labels = list(labels)

print("\nClasses:")
for i, label in enumerate(labels):
    print(f"  {i}: {label}")


# ============================================================
# FEATURE EXTRACTION FROM CSV
# ============================================================

def get_xy(df):
    """
    Extracts 126 landmark features and integer labels.

    The preparation script stores the label in a 'label' column
    and the remaining numeric columns are the 126 features.
    """

    if "label" not in df.columns:
        raise ValueError(
            "CSV does not contain a 'label' column."
        )

    feature_columns = [
        col for col in df.columns
        if col != "label"
        and col not in ("image_path", "path", "filepath")
    ]

    # Keep only numeric feature columns
    numeric_columns = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    X = df[numeric_columns].values.astype(np.float32)

    # Handle either string labels or integer labels
    if pd.api.types.is_numeric_dtype(df["label"]):
        y = df["label"].values.astype(np.int64)
    else:
        label_to_index = {
            label: i for i, label in enumerate(labels)
        }

        y = np.array(
            [label_to_index[str(x)] for x in df["label"]],
            dtype=np.int64
        )

    return X, y


X_train, y_train = get_xy(train_df)
X_test, y_test = get_xy(test_df)
X_val, y_val = get_xy(val_df)


# ============================================================
# CHECK FEATURES
# ============================================================

print("\nFeature shape:")
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("X_val:  ", X_val.shape)

if X_train.shape[1] != 126:
    raise ValueError(
        f"Expected 126 features, but found {X_train.shape[1]}."
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=len(labels)
)

total = len(y_train)
num_classes = len(labels)

class_weights = {}

for class_index, count in enumerate(class_counts):
    if count > 0:
        class_weights[class_index] = (
            total / (num_classes * count)
        )

print("\nClass distribution:")

for i, label in enumerate(labels):
    print(
        f"  {label}: {class_counts[i]} "
        f"(weight={class_weights.get(i, 1.0):.3f})"
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding neural network...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(126,)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(
        num_classes,
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
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
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

print("\nStarting training...\n")

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

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST
# ============================================================

print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\nPer-class accuracy:")

predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

for class_index, label in enumerate(labels):

    mask = y_test == class_index

    if np.sum(mask) == 0:
        continue

    accuracy = np.mean(
        predicted_classes[mask] == class_index
    )

    print(
        f"  {label}: "
        f"{accuracy * 100:.2f}% "
        f"({np.sum(mask)} samples)"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

with open(LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump(
        labels,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nModel saved to:")
print(MODEL_PATH)

print(f"\nLabels saved to:")
print(LABELS_OUT)

print("\nYou can now run:")
print("python .\\test_correction_live.py")