"""
Train a small LSTM/dense model on recorded word landmarks.

    python tools/train_word_model.py --epochs 35 --batch-size 16
    python tools/train_word_model.py --generate-synthetic 10

Writes app/models/signmate_word_model.keras and word_labels.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    print("Error: TensorFlow is required. Run: pip install tensorflow")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = BASE_DIR / "dataset" / "words" / "landmarks"
MODELS_DIR = BASE_DIR / "app" / "models"
METADATA_PATH = BASE_DIR / "dataset" / "words" / "words_metadata.json"


def generate_synthetic_samples(words, samples_per_word=10, frames=30):
    """
    Fake landmark sequences so you can test training without a webcam.
    """
    print(f"\nGenerating {samples_per_word} synthetic samples for each of {len(words)} words...")
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    for word in words:
        slug = word.strip().upper().replace(" ", "_")
        target_dir = LANDMARKS_DIR / slug
        target_dir.mkdir(parents=True, exist_ok=True)

        for s in range(1, samples_per_word + 1):
            out_file = target_dir / f"sample_{s:03d}.npy"
            if out_file.exists():
                continue

            # Base anchor + smooth temporal sinusoidal trajectory + small gaussian noise
            base = np.random.uniform(0.2, 0.8, size=(126,)).astype(np.float32)
            time_steps = np.linspace(0, np.pi, frames).reshape(frames, 1)
            motion = np.sin(time_steps) * np.random.uniform(-0.15, 0.15, size=(1, 126))
            noise = np.random.normal(0, 0.015, size=(frames, 126)).astype(np.float32)
            sample_data = np.clip(base + motion + noise, 0.0, 1.0)

            np.save(str(out_file), sample_data)

    print("Synthetic sample generation completed!\n")


def load_dataset(max_frames=30):
    if not LANDMARKS_DIR.exists():
        return [], [], []

    classes = sorted([d.name for d in LANDMARKS_DIR.iterdir() if d.is_dir()])
    if not classes:
        return [], [], []

    x_data = []
    y_data = []

    print(f"Loading landmark sequences from {LANDMARKS_DIR}...")
    for idx, class_name in enumerate(classes):
        class_dir = LANDMARKS_DIR / class_name
        files = list(class_dir.glob("sample_*.npy"))
        for f in files:
            seq = np.load(str(f))
            # Standardize length to max_frames
            if len(seq) < max_frames:
                # Pad with final frame
                pad = np.repeat(seq[-1:], max_frames - len(seq), axis=0)
                seq = np.vstack([seq, pad])
            elif len(seq) > max_frames:
                # Truncate
                seq = seq[:max_frames]

            x_data.append(seq)
            y_data.append(idx)

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int32), classes


def build_word_model(num_classes: int, sequence_length: int = 30, num_features: int = 126):
    model = models.Sequential([
        layers.Input(shape=(sequence_length, num_features)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="SignMate ISL Word Classifier Trainer")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--generate-synthetic", type=int, default=0, help="Generate N synthetic samples per class for pipeline verification")

    args = parser.parse_args()

    # Read standard words
    target_words = ["NAMASTE", "HELLO", "THANK_YOU", "PLEASE", "GOOD", "HELP", "YES", "NO", "WATER", "EAT"]
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            target_words = [w["word"].replace(" ", "_") for w in json.load(f).get("words", [])]

    if args.generate_synthetic > 0:
        generate_synthetic_samples(target_words, samples_per_word=args.generate_synthetic)

    X, y, classes = load_dataset()

    if len(classes) == 0 or len(X) == 0:
        print("\nNo landmark data found in dataset/words/landmarks/.")
        print("To generate verification samples and test the pipeline, run:")
        print("    python tools/train_word_model.py --generate-synthetic 10")
        print("\nOr record real gestures via webcam using:")
        print("    python tools/record_word_dataset.py --word NAMASTE --samples 20\n")
        return

    print(f"\nLoaded {len(X)} samples across {len(classes)} classes.")
    print(f"Input shape per sample: {X[0].shape} (30 frames x 126 coordinates)")

    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Train / Val Split (80 / 20)
    split_idx = int(0.8 * len(X))
    x_train, x_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Training set: {len(x_train)} samples | Validation set: {len(x_val)} samples")

    model = build_word_model(num_classes=len(classes))
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_model_path = MODELS_DIR / "signmate_word_model.keras"
    out_labels_path = MODELS_DIR / "word_labels.json"

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    )

    print(f"\nTraining model for up to {args.epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val) if len(x_val) > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop] if len(x_val) > 0 else []
    )

    # Save model and label mapping
    model.save(str(out_model_path))
    with open(out_labels_path, "w", encoding="utf-8") as lf:
        json.dump({
            "classes": classes,
            "num_classes": len(classes),
            "input_shape": [30, 126]
        }, lf, indent=2)

    print(f"\nTrained word model successfully saved to: {out_model_path}")
    print(f"Word labels saved to: {out_labels_path}")
    print(f"Model ready for deployment!\n")


if __name__ == "__main__":
    main()
