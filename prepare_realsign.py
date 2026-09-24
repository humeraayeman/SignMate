from pathlib import Path
import csv
import json

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "RealSign"

OUTPUT_DIR = BASE_DIR / "ml" / "data" / "realsign"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_OUTPUT = OUTPUT_DIR / "train.csv"
TEST_OUTPUT = OUTPUT_DIR / "test.csv"
VAL_OUTPUT = OUTPUT_DIR / "validation.csv"

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

MAX_TRAIN_PER_CLASS = 700
MAX_TEST_PER_CLASS = 200
MAX_VAL_PER_CLASS = 100


# ============================================================
# MEDIAPIPE
# ============================================================

print()
print("=" * 60)
print("SIGN SYNC - REALSIGN DATA PREPARATION")
print("=" * 60)
print()

print("Loading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.45,
    min_hand_presence_confidence=0.45,
    min_tracking_confidence=0.45
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe loaded successfully.")


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(landmarks):

    wrist = landmarks[0]

    points = []

    for landmark in landmarks:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        points.append([
            x,
            y,
            z
        ])

    points = np.array(
        points,
        dtype=np.float32
    )

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])

    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    scale = max(
        width,
        height
    )

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(
        mp_image
    )

    detected_hands = result.hand_landmarks

    if not detected_hands:
        return None

    hands = []

    for hand in detected_hands:

        features = normalize_hand(hand)

        wrist_x = hand[0].x

        hands.append({
            "x": wrist_x,
            "features": features
        })

    hands.sort(
        key=lambda h: h["x"]
    )

    # One hand
    if len(hands) == 1:

        first = hands[0]["features"]

        second = np.zeros(
            63,
            dtype=np.float32
        )

        return np.concatenate([
            first,
            second
        ])

    # Two hands
    first = hands[0]["features"]

    second = hands[1]["features"]

    return np.concatenate([
        first,
        second
    ])


# ============================================================
# FIND CLASSES
# ============================================================

def get_classes(folder):

    classes = []

    if not folder.exists():
        return classes

    for item in folder.iterdir():

        if item.is_dir():

            name = item.name.strip()

            if len(name) == 1 and name.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

                classes.append(
                    name.upper()
                )

    return sorted(
        set(classes)
    )


classes = get_classes(
    DATASET_DIR / "Training"
)

print()
print("Classes found:")
print(classes)
print()
print("Number of classes:", len(classes))


if len(classes) != 26:

    print()
    print(
        "WARNING: Expected 26 alphabet classes."
    )
    print(
        "Please check the Training folder."
    )


# ============================================================
# CSV WRITER
# ============================================================

def create_writer(path):

    file = open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(file)

    header = [
        "label"
    ]

    for i in range(126):

        header.append(
            f"feature_{i + 1}"
        )

    writer.writerow(header)

    return file, writer


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name,
    max_per_class,
    output_file
):

    input_dir = (
        DATASET_DIR
        / split_name
    )

    if not input_dir.exists():

        print(
            f"Skipping {split_name}: folder not found."
        )

        return

    print()
    print("=" * 60)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 60)

    file, writer = create_writer(
        output_file
    )

    total_images = 0
    processed_images = 0
    failed_images = 0

    try:

        for label in classes:

            class_dir = (
                input_dir
                / label
            )

            if not class_dir.exists():

                print(
                    f"{label}: folder missing"
                )

                continue

            images = []

            for image_path in class_dir.iterdir():

                if image_path.is_file():

                    if image_path.suffix.lower() in IMAGE_EXTENSIONS:

                        images.append(
                            image_path
                        )

            images = sorted(
                images
            )

            images = images[
                :max_per_class
            ]

            class_processed = 0
            class_failed = 0

            for image_path in images:

                total_images += 1

                frame = cv2.imread(
                    str(image_path)
                )

                if frame is None:

                    failed_images += 1
                    class_failed += 1
                    continue

                features = extract_features(
                    frame
                )

                if features is None:

                    failed_images += 1
                    class_failed += 1
                    continue

                writer.writerow(
                    [label]
                    + features.tolist()
                )

                processed_images += 1
                class_processed += 1

            file.flush()

            print(
                f"{label}: "
                f"{class_processed} processed, "
                f"{class_failed} failed"
            )

    finally:

        file.close()

    print()
    print(
        f"{split_name} finished."
    )
    print(
        f"Images checked: {total_images}"
    )
    print(
        f"Processed: {processed_images}"
    )
    print(
        f"Failed: {failed_images}"
    )


# ============================================================
# PROCESS ALL DATA
# ============================================================

try:

    process_split(
        "Training",
        MAX_TRAIN_PER_CLASS,
        TRAIN_OUTPUT
    )

    process_split(
        "Testing",
        MAX_TEST_PER_CLASS,
        TEST_OUTPUT
    )

    process_split(
        "Validation",
        MAX_VAL_PER_CLASS,
        VAL_OUTPUT
    )

finally:

    hand_landmarker.close()


# ============================================================
# SAVE LABELS
# ============================================================

labels_file = (
    OUTPUT_DIR
    / "labels.json"
)

with open(
    labels_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        classes,
        f,
        indent=2
    )


print()
print("=" * 60)
print("PREPARATION COMPLETE")
print("=" * 60)
print()
print("Output directory:")
print(OUTPUT_DIR)
print()
print("Created:")
print("  train.csv")
print("  test.csv")
print("  validation.csv")
print("  labels.json")
print()
print("Next step: train the classifier.")
print("=" * 60)