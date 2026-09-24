"""
SignMate - Indian Sign Language Word Image Classifier Trainer
=============================================================
Trains a deep neural network on MediaPipe landmark features extracted
from the images in dataset/words_dataset/<WORD>/*.jpg.

Usage:
    python train_words_model.py --epochs 30 --batch-size 32

Outputs:
    ml/trained/signsync_words.keras (and app/models/signsync_words.keras)
    ml/trained/labels_words.json (and app/models/labels_words.json)
    ml/trained/words_normalization.npz (and app/models/words_normalization.npz)
"""

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ml" / "data" / "words"
OUTPUT_DIR = BASE_DIR / "ml" / "trained"
APP_MODELS_DIR = BASE_DIR / "app" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
VAL_FILE = DATA_DIR / "validation.csv"

MODEL_FILE = OUTPUT_DIR / "signsync_words.keras"
LABELS_FILE = OUTPUT_DIR / "labels_words.json"
NORM_FILE = OUTPUT_DIR / "words_normalization.npz"


def main():
    parser = argparse.ArgumentParser(description="SignMate ISL Word Model Trainer")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" SIGNMATE - WORDS MODEL TRAINING (IMAGE FEATURES)")
    print("=" * 65 + "\n")

    if not TRAIN_FILE.exists():
        print(f"Error: Training file not found at {TRAIN_FILE}.")
        print("Please run 'python prepare_words_dataset.py' first.\n")
        return

    print("Loading image landmark datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    val_df = pd.read_csv(VAL_FILE)

    print(f"  Training samples:   {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Testing samples:    {len(test_df)}")

    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    print(f"  Number of features: {len(feature_cols)} (126 coordinates)\n")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df["label"].values

    X_val = val_df[feature_cols].values.astype(np.float32) if len(val_df) else None
    y_val_raw = val_df["label"].values if len(val_df) else None

    X_test = test_df[feature_cols].values.astype(np.float32) if len(test_df) else None
    y_test_raw = test_df["label"].values if len(test_df) else None

    # Classes
    labels = sorted(np.unique(y_train_raw).tolist())
    print(f"Detected {len(labels)} word classes:")
    print(", ".join(labels[:10]) + ("..." if len(labels) > 10 else ""))
    print()

    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_train = np.array([label_to_idx[l] for l in y_train_raw], dtype=np.int32)
    y_val = np.array([label_to_idx[l] for l in y_val_raw], dtype=np.int32) if y_val_raw is not None else None
    y_test = np.array([label_to_idx[l] for l in y_test_raw], dtype=np.int32) if y_test_raw is not None else None

    # Feature normalization
    print("Computing feature normalization (mean and standard deviation)...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    if X_val is not None:
        X_val = (X_val - mean) / std
    if X_test is not None:
        X_test = (X_test - mean) / std

    # Build model (matches RealSign architecture for consistency)
    print("Building neural network architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(feature_cols),)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(len(labels), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if X_val is not None else "loss",
            patience=8,
            mode="max" if X_val is not None else "min",
            restore_best_weights=True
        )
    ]

    print(f"\nTraining model for up to {args.epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    if X_test is not None:
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n[Test Set Performance] Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    # Save model and artifacts to both ml/trained/ and app/models/
    print("\nSaving trained model and artifacts...")
    for target_dir in [OUTPUT_DIR, APP_MODELS_DIR]:
        model.save(str(target_dir / "signsync_words.keras"))
        with open(target_dir / "labels_words.json", "w", encoding="utf-8") as lf:
            json.dump({
                "classes": labels,
                "label_to_index": label_to_idx,
                "num_classes": len(labels)
            }, lf, indent=2)
        np.savez(target_dir / "words_normalization.npz", mean=mean, std=std)

    print(f"  [OK] Model saved to {MODEL_FILE}")
    print(f"  [OK] Labels saved to {LABELS_FILE}")
    print(f"  [OK] Normalization saved to {NORM_FILE}")
    print("\nTraining completed successfully! Model ready for inference.\n")


if __name__ == "__main__":
    main()
