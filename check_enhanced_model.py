from pathlib import Path
import json
import csv

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml" / "data" / "realsign_enhanced"

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "signsync_realsign_enhanced.keras"
)

LABELS_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "labels_realsign_enhanced.json"
)

NORM_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "realsign_enhanced_normalization.npz"
)

TARGETS = ["I", "O", "R", "T", "V", "Y"]


def load_csv(path):

    X = []
    y = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.reader(f)

        next(reader)

        for row in reader:

            y.append(row[0])

            X.append([
                float(x)
                for x in row[1:]
            ])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y)
    )


print()
print("=" * 60)
print("CHECKING ENHANCED MODEL")
print("=" * 60)
print()

print("Loading labels...")

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:

    labels = json.load(f)


label_to_index = {
    label: i
    for i, label in enumerate(labels)
}


print("Loading test dataset...")

X, y_text = load_csv(
    DATA_DIR / "test.csv"
)


print("Loading normalization...")

normalization = np.load(
    NORM_PATH
)

mean = normalization["mean"]
std = normalization["std"]

std[std < 1e-6] = 1.0


X = (
    X - mean
) / std


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("Running predictions...")

predictions = model.predict(
    X,
    verbose=0
)

predicted = np.argmax(
    predictions,
    axis=1
)


print()
print("=" * 60)
print("TARGET SIGN ACCURACY")
print("=" * 60)
print()


for label in TARGETS:

    index = label_to_index[label]

    mask = (
        y_text == label
    )

    total = int(np.sum(mask))

    correct = int(
        np.sum(
            predicted[mask] == index
        )
    )

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{label}: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )


print()
print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)
print()