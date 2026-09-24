from pathlib import Path
import json
import random

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "Prathum_ISL" / "Indian"
OUTPUT_DIR = BASE_DIR / "ml" / "data" / "jksy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGETS = ["J", "K", "S", "Y"]

TRAIN_PER_CLASS = 1200
TEST_PER_CLASS = 300
VAL_PER_CLASS = 200

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hands = []

    for hand in result.hand_landmarks:

        wrist = hand[0]

        points = np.array(
            [
                [
                    p.x - wrist.x,
                    p.y - wrist.y,
                    p.z - wrist.z
                ]
                for p in hand
            ],
            dtype=np.float32
        )

        radial = np.sqrt(
            points[:, 0] ** 2 +
            points[:, 1] ** 2
        )

        scale = float(np.max(radial))

        if scale < 1e-6:
            scale = 1.0

        points /= scale

        hands.append(
            (
                wrist.x,
                points.flatten()
            )
        )

    hands.sort(key=lambda x: x[0])

    first = hands[0][1]

    if len(hands) >= 2:
        second = hands[1][1]
    else:
        second = np.zeros(63, dtype=np.float32)

    return np.concatenate([first, second])


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(letter):

    folder = DATASET_DIR / letter

    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        return []

    files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(folder.rglob(ext))

    return files


# ============================================================
# PROCESS
# ============================================================

data = {
    "train": [],
    "test": [],
    "validation": []
}

labels = []

for label_index, letter in enumerate(TARGETS):

    print()
    print("=" * 60)
    print(f"PROCESSING {letter}")
    print("=" * 60)

    images = find_images(letter)

    print("Images found:", len(images))

    random.shuffle(images)

    required = (
        TRAIN_PER_CLASS +
        TEST_PER_CLASS +
        VAL_PER_CLASS
    )

    if len(images) < required:
        print(
            f"WARNING: {letter} has only {len(images)} images, "
            f"but {required} were requested."
        )

    images = images[:required]

    train_files = images[:TRAIN_PER_CLASS]

    test_start = TRAIN_PER_CLASS
    test_end = TRAIN_PER_CLASS + TEST_PER_CLASS

    test_files = images[test_start:test_end]

    val_files = images[test_end:test_end + VAL_PER_CLASS]

    for split_name, files in [
        ("train", train_files),
        ("test", test_files),
        ("validation", val_files)
    ]:

        print(f"{split_name}: {len(files)}")

        for image_path in files:

            features = extract_features(image_path)

            if features is None:
                continue

            data[split_name].append(
                {
                    "features": features.tolist(),
                    "label": label_index
                }
            )

    labels.append(letter)


# ============================================================
# SAVE
# ============================================================

for split_name in data:

    output_file = OUTPUT_DIR / f"{split_name}.npz"

    X = np.array(
        [item["features"] for item in data[split_name]],
        dtype=np.float32
    )

    y = np.array(
        [item["label"] for item in data[split_name]],
        dtype=np.int64
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print()
    print(
        f"{split_name}: "
        f"X={X.shape}, y={y.shape}"
    )


labels_file = OUTPUT_DIR / "labels.json"

with open(labels_file, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)


print()
print("=" * 60)
print("J K S Y PREPARATION COMPLETE")
print("=" * 60)
print("Labels:", labels)
print("Output:", OUTPUT_DIR)
print("=" * 60)

landmarker.close()