import csv
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "RealSign-Extracted"

OUTPUT_DIR = BASE_DIR / "ml" / "data"

OUTPUT_FILE = OUTPUT_DIR / "landmarks_normalized.csv"

MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


if not DATASET_DIR.exists():
    print("ERROR: Dataset folder not found!")
    print(DATASET_DIR)
    exit()


if not MODEL_PATH.exists():
    print("ERROR: MediaPipe model not found!")
    print(MODEL_PATH)
    exit()


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# CSV HEADER
# ============================================================

header = ["label"]

for i in range(21):
    header.append(f"hand1_x{i}")

for i in range(21):
    header.append(f"hand1_y{i}")

for i in range(21):
    header.append(f"hand1_z{i}")

for i in range(21):
    header.append(f"hand2_x{i}")

for i in range(21):
    header.append(f"hand2_y{i}")

for i in range(21):
    header.append(f"hand2_z{i}")


# ============================================================
# IMAGE FILES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = []

for file in DATASET_DIR.rglob("*"):
    if (
        file.is_file()
        and file.suffix.lower() in image_extensions
    ):
        image_files.append(file)


print()
print("==============================================")
print(" SignSync - Improved Dataset Preparation")
print("==============================================")
print()

print("Images found:", len(image_files))
print()


if not image_files:
    print("ERROR: No images found.")
    detector.close()
    exit()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_hand_features(hand):

    if hand is None:
        return [0.0] * 63

    wrist = hand[0]

    features = []

    # X
    for landmark in hand:
        features.append(
            landmark.x - wrist.x
        )

    # Y
    for landmark in hand:
        features.append(
            landmark.y - wrist.y
        )

    # Z
    for landmark in hand:
        features.append(
            landmark.z - wrist.z
        )

    return features


# ============================================================
# SORT HANDS
# ============================================================

def get_hand_features(result):

    hands = []

    if not result.hand_landmarks:
        return None, None

    for hand in result.hand_landmarks:

        features = extract_hand_features(hand)

        # Calculate approximate hand size.
        # This is used to normalize the hand.
        xs = [landmark.x for landmark in hand]
        ys = [landmark.y for landmark in hand]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        size = max(width, height, 0.001)

        normalized = []

        for i in range(63):

            value = features[i]

            normalized.append(
                value / size
            )

        hands.append(normalized)

    # --------------------------------------------------------
    # ONE HAND
    # --------------------------------------------------------

    if len(hands) == 1:

        return hands[0], [0.0] * 63


    # --------------------------------------------------------
    # TWO HANDS
    # --------------------------------------------------------

    if len(hands) >= 2:

        # Sort hands consistently using wrist X position.
        #
        # This avoids depending completely on MediaPipe's
        # Left/Right classification.
        #
        # The hand appearing on the left side of the image
        # becomes hand1.
        #
        # The hand appearing on the right side becomes hand2.

        hand_info = []

        for index, hand in enumerate(
            result.hand_landmarks[:2]
        ):

            wrist_x = hand[0].x

            hand_info.append(
                (wrist_x, hands[index])
            )

        hand_info.sort(
            key=lambda item: item[0]
        )

        hand1 = hand_info[0][1]
        hand2 = hand_info[1][1]

        return hand1, hand2


    return None, None


# ============================================================
# PROCESS DATASET
# ============================================================

processed = 0
failed = 0

one_hand_count = 0
two_hand_count = 0


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow(header)

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        label = image_path.parent.name

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            failed += 1
            continue


        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )


        try:

            result = detector.detect(
                mp_image
            )

        except Exception:

            failed += 1
            continue


        if not result.hand_landmarks:

            failed += 1
            continue


        hand1, hand2 = get_hand_features(
            result
        )


        if hand1 is None:

            failed += 1
            continue


        if sum(abs(x) for x in hand2) == 0:

            one_hand_count += 1

        else:

            two_hand_count += 1


        row = (
            [label]
            + hand1
            + hand2
        )


        writer.writerow(row)

        processed += 1


        if index % 250 == 0:

            print(
                f"Processed {index}/{len(image_files)} "
                f"| One-hand: {one_hand_count} "
                f"| Two-hand: {two_hand_count}"
            )


# ============================================================
# FINISH
# ============================================================

detector.close()


print()
print("==============================================")
print(" Dataset preparation completed")
print("==============================================")
print()

print("Total images :", len(image_files))
print("Processed    :", processed)
print("Failed       :", failed)

print()

print("One-hand samples :", one_hand_count)
print("Two-hand samples :", two_hand_count)

print()

print("Features per sample:", 126)

print()

print("CSV:")
print(OUTPUT_FILE)

print()