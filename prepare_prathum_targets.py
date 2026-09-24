from pathlib import Path
import json
import csv
import random

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "dataset" / "Prathum_ISL" / "Indian"
OUTPUT_DIR = BASE_DIR / "ml" / "data" / "prathum_targets"
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

TARGETS = ["I", "O", "R", "T", "V", "Y"]

TRAIN_PER_CLASS = 800
TEST_PER_CLASS = 250
VAL_PER_CLASS = 150

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# NORMALIZATION
# Same 126-feature idea used by SignSync
# ============================================================

def normalize_hand(hand):
    """
    Convert 21 MediaPipe landmarks into 63 normalized values.

    - wrist becomes origin
    - scale normalized using maximum distance from wrist
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    wrist = points[0].copy()

    points = points - wrist

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


def extract_features(image, detector):
    """
    Detect up to two hands.

    Hands are sorted by wrist X position.
    One-hand signs are zero padded to 126 features.
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hands = list(result.hand_landmarks)

    # Sort hands consistently from left to right.
    hands.sort(key=lambda h: h[0].x)

    features = []

    for hand in hands[:2]:
        features.extend(normalize_hand(hand))

    # 63 features per hand.
    # Pad second hand if only one hand exists.
    while len(features) < 126:
        features.extend([0.0] * 63)

    return features[:126]


# ============================================================
# DATASET PREPARATION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = []
    test_rows = []
    val_rows = []

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2
    )

    processed = 0
    failed = 0

    print()
    print("=" * 60)
    print("Preparing Prathum ISL target dataset")
    print("=" * 60)
    print()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

        for label in TARGETS:

            folder = SOURCE_DIR / label

            if not folder.exists():
                print(f"[ERROR] Missing folder: {folder}")
                continue

            images = [
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                }
            ]

            random.shuffle(images)

            # Limit to enough images for our split.
            required = TRAIN_PER_CLASS + TEST_PER_CLASS + VAL_PER_CLASS
            images = images[:required]

            train_images = images[:TRAIN_PER_CLASS]
            test_images = images[
                TRAIN_PER_CLASS:
                TRAIN_PER_CLASS + TEST_PER_CLASS
            ]
            val_images = images[
                TRAIN_PER_CLASS + TEST_PER_CLASS:
                TRAIN_PER_CLASS + TEST_PER_CLASS + VAL_PER_CLASS
            ]

            print(
                f"{label}: "
                f"{len(train_images)} train, "
                f"{len(test_images)} test, "
                f"{len(val_images)} validation"
            )

            def process_images(image_list, destination):

                nonlocal processed, failed

                for image_path in image_list:

                    image = cv2.imread(str(image_path))

                    if image is None:
                        failed += 1
                        continue

                    features = extract_features(image, detector)

                    if features is None:
                        failed += 1
                        continue

                    row = features + [label]
                    destination.append(row)

                    processed += 1

            process_images(train_images, train_rows)
            process_images(test_images, test_rows)
            process_images(val_images, val_rows)

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    header = [f"f{i}" for i in range(126)] + ["label"]

    def save_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    save_csv(OUTPUT_DIR / "train.csv", train_rows)
    save_csv(OUTPUT_DIR / "test.csv", test_rows)
    save_csv(OUTPUT_DIR / "validation.csv", val_rows)

    # ========================================================
    # LABELS
    # ========================================================

    with open(
        OUTPUT_DIR / "labels.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(TARGETS, f, indent=2)

    print()
    print("=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Failed:    {failed}")
    print()
    print(f"Train rows:      {len(train_rows)}")
    print(f"Test rows:       {len(test_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    print()
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()