from pathlib import Path
import json
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "Prathum_ISL" / "Indian"
OUT_DIR = BASE_DIR / "ml" / "data" / "prathum_ck"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["C", "K"]

TRAIN_PER_CLASS = 900
TEST_PER_CLASS = 200
VAL_PER_CLASS = 100

hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(BASE_DIR / "models" / "hand_landmarker.task")
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2
    )
)


def extract_features(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(mp_image)

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
            (wrist.x, points.flatten())
        )

    hands.sort(key=lambda x: x[0])

    first = hands[0][1]

    if len(hands) >= 2:
        second = hands[1][1]
    else:
        second = np.zeros(63, dtype=np.float32)

    return np.concatenate([first, second])


def process_split(split_name, count_per_class):
    X = []
    y = []

    for class_index, label in enumerate(CLASSES):
        class_dir = DATASET_DIR / label

        images = sorted(
            [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        if split_name == "train":
            images = images[:count_per_class]
        elif split_name == "test":
            start = TRAIN_PER_CLASS
            images = images[start:start + count_per_class]
        else:
            start = TRAIN_PER_CLASS + TEST_PER_CLASS
            images = images[start:start + count_per_class]

        print(f"{split_name}: {label} -> {len(images)} images")

        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            features = extract_features(image)

            if features is not None:
                X.append(features)
                y.append(class_index)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.save(OUT_DIR / f"X_{split_name}.npy", X)
    np.save(OUT_DIR / f"y_{split_name}.npy", y)

    print(
        f"{split_name}: X={X.shape}, y={y.shape}"
    )


process_split("train", TRAIN_PER_CLASS)
process_split("test", TEST_PER_CLASS)
process_split("val", VAL_PER_CLASS)

with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
    json.dump(CLASSES, f)

print("\nC/K DATA PREPARATION COMPLETE")