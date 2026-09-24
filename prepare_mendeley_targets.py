from pathlib import Path
import json
import csv
import random

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MENDELEY_ROOT = (
    BASE_DIR
    / "dataset"
    / "ISL Hand Gesture Dataset"
)

REALSIGN_ROOT = BASE_DIR / "ml" / "data" / "realsign"

OUTPUT_ROOT = BASE_DIR / "ml" / "data" / "realsign_enhanced"

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

TARGET_CLASSES = ["I", "O", "R", "T", "V", "Y"]

ALL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# Maximum external images per target class.
# Keeping this moderate makes MediaPipe processing reasonable.
MAX_EXTERNAL_PER_CLASS = 500

# Number of existing RealSign target images to keep.
# 0 means external data replaces the target classes in the
# enhanced training set.
KEEP_REALSIGN_TARGET_TRAIN = 300

random.seed(RANDOM_SEED)


# ============================================================
# FIND IMAGE FOLDERS AUTOMATICALLY
# ============================================================

def find_class_images(root):
    """
    Searches recursively for directories whose names are
    single A-Z class labels and contain image files.
    """

    candidates = {}

    for directory in root.rglob("*"):

        if not directory.is_dir():
            continue

        folder_name = directory.name.strip().upper()

        if folder_name not in ALL_CLASSES:
            continue

        images = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            continue

        candidates.setdefault(folder_name, [])

        candidates[folder_name].extend(images)

    # Remove duplicates.
    for label in candidates:
        candidates[label] = sorted(
            set(candidates[label]),
            key=lambda p: str(p)
        )

    return candidates


# ============================================================
# MEDIAPIPE
# ============================================================

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{HAND_MODEL_PATH}"
    )

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40,
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# LANDMARK FEATURE EXTRACTION
# ============================================================

def normalize_hand(landmarks):

    wrist = landmarks[0]

    points = []

    for landmark in landmarks:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        points.append([x, y, z])

    points = np.array(points, dtype=np.float32)

    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])

    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    scale = max(width, height)

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


def extract_features(image_path):

    frame = cv2.imread(str(image_path))

    if frame is None:
        return None

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(mp_image)

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

    hands.sort(key=lambda h: h["x"])

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

    first = hands[0]["features"]
    second = hands[1]["features"]

    return np.concatenate([
        first,
        second
    ])


# ============================================================
# CSV WRITER
# ============================================================

def save_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["label"] + [
                f"f{i}"
                for i in range(126)
            ]
        )

        for label, features in rows:

            writer.writerow(
                [label] + features.tolist()
            )


# ============================================================
# LOAD EXISTING REALSIGN DATA
# ============================================================

def load_realsign_csv(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.reader(f)

        header = next(reader)

        for row in reader:

            label = row[0]

            features = np.array(
                [float(x) for x in row[1:]],
                dtype=np.float32
            )

            rows.append(
                (label, features)
            )

    return rows


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("SIGN SYNC - MENDELEY TARGET DATA PREPARATION")
print("=" * 70)
print()

print("Searching Mendeley dataset...")
print(MENDELEY_ROOT)

class_images = find_class_images(
    MENDELEY_ROOT
)

print()
print("Detected image classes:")

for label in ALL_CLASSES:

    count = len(
        class_images.get(label, [])
    )

    if count:
        print(
            f"  {label}: {count} images"
        )

print()


# ============================================================
# VERIFY TARGET CLASSES
# ============================================================

missing = [
    label
    for label in TARGET_CLASSES
    if label not in class_images
]

if missing:

    print(
        "WARNING: These target classes were not found:"
    )

    print(
        ", ".join(missing)
    )

    print()
    print(
        "No files were changed."
    )

    raise SystemExit(1)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD EXISTING REALSIGN DATA
# ============================================================

print("Loading existing RealSign data...")

realsign_train = load_realsign_csv(
    REALSIGN_ROOT / "train.csv"
)

realsign_test = load_realsign_csv(
    REALSIGN_ROOT / "test.csv"
)

realsign_validation = load_realsign_csv(
    REALSIGN_ROOT / "validation.csv"
)

print(
    "Existing training rows:",
    len(realsign_train)
)

print(
    "Existing testing rows:",
    len(realsign_test)
)

print(
    "Existing validation rows:",
    len(realsign_validation)
)


# ============================================================
# KEEP NON-TARGET CLASSES EXACTLY AS THEY ARE
# ============================================================

enhanced_train = [
    row
    for row in realsign_train
    if row[0] not in TARGET_CLASSES
]

enhanced_test = [
    row
    for row in realsign_test
    if row[0] not in TARGET_CLASSES
]

enhanced_validation = [
    row
    for row in realsign_validation
    if row[0] not in TARGET_CLASSES
]


# ============================================================
# OPTIONALLY KEEP SOME REALSIGN TARGET DATA
# ============================================================

for label in TARGET_CLASSES:

    target_rows = [
        row
        for row in realsign_train
        if row[0] == label
    ]

    random.shuffle(target_rows)

    if KEEP_REALSIGN_TARGET_TRAIN > 0:

        enhanced_train.extend(
            target_rows[
                :KEEP_REALSIGN_TARGET_TRAIN
            ]
        )


# ============================================================
# PROCESS MENDELEY TARGET CLASSES
# ============================================================

external_train = []
external_test = []
external_validation = []

print()
print("=" * 70)
print("PROCESSING MENDELEY TARGET CLASSES")
print("=" * 70)

for label in TARGET_CLASSES:

    images = class_images[label].copy()

    random.shuffle(images)

    images = images[
        :MAX_EXTERNAL_PER_CLASS
    ]

    print()
    print(
        f"{label}: processing {len(images)} images..."
    )

    successful = 0
    failed = 0

    extracted = []

    for index, image_path in enumerate(images, start=1):

        features = extract_features(
            image_path
        )

        if features is None:

            failed += 1
            continue

        extracted.append(
            (
                label,
                features
            )
        )

        successful += 1

        if index % 50 == 0:

            print(
                f"  {index}/{len(images)} checked..."
            )

    print(
        f"{label}: successful={successful}, failed={failed}"
    )

    # Shuffle extracted landmark data.
    random.shuffle(extracted)

    total = len(extracted)

    train_end = int(total * 0.70)
    test_end = int(total * 0.85)

    external_train.extend(
        extracted[:train_end]
    )

    external_test.extend(
        extracted[train_end:test_end]
    )

    external_validation.extend(
        extracted[test_end:]
    )


# ============================================================
# COMBINE
# ============================================================

enhanced_train.extend(
    external_train
)

enhanced_test.extend(
    external_test
)

enhanced_validation.extend(
    external_validation
)


random.shuffle(enhanced_train)
random.shuffle(enhanced_test)
random.shuffle(enhanced_validation)


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 70)
print("SAVING ENHANCED DATASET")
print("=" * 70)

save_csv(
    OUTPUT_ROOT / "train.csv",
    enhanced_train
)

save_csv(
    OUTPUT_ROOT / "test.csv",
    enhanced_test
)

save_csv(
    OUTPUT_ROOT / "validation.csv",
    enhanced_validation
)


with open(
    OUTPUT_ROOT / "labels.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        ALL_CLASSES,
        f,
        indent=2
    )


print()
print("ENHANCED DATASET CREATED")
print()
print(
    "Training:",
    len(enhanced_train)
)

print(
    "Testing:",
    len(enhanced_test)
)

print(
    "Validation:",
    len(enhanced_validation)
)

print()
print("Output:")
print(OUTPUT_ROOT)

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print("Your original RealSign dataset was NOT modified.")
print("Your original trained model was NOT modified.")
print("Only the new enhanced dataset was created.")
print()