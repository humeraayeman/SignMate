import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

OUTPUT_DIR = BASE_DIR / "ml" / "data"

OUTPUT_FILE = OUTPUT_DIR / "webcam_calibration.csv"

SAMPLES_PER_LETTER = 20

CAPTURE_DELAY = 0.15

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("Loading MediaPipe hand detector...")

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

    scale = max(width, height)

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


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


print()
print("Opening webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print()
    print("ERROR: Could not open webcam.")
    print("Check that your camera is not being used by another application.")
    exit()


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


file_exists = OUTPUT_FILE.exists()

csv_file = open(
    OUTPUT_FILE,
    "a",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)


if not file_exists:

    header = ["label"]

    for i in range(126):
        header.append(f"feature_{i + 1}")

    writer.writerow(header)


print()
print("=" * 60)
print("SIGN SYNC WEBCAM CALIBRATION")
print("=" * 60)
print()
print("You will collect samples for A-Z.")
print()
print("For each letter:")
print("1. Make the correct sign.")
print("2. Press the matching keyboard key.")
print("3. Wait for the countdown.")
print("4. Keep your hand visible while samples are captured.")
print()
print("20 samples will be collected for each letter.")
print()
print("Press Q at any time to quit.")
print()
print("=" * 60)


try:

    for letter in LETTERS:

        print()
        print("-" * 60)
        print(f"GET READY: {letter}")
        print(f"Press [{letter}] when your hand is ready.")
        print("-" * 60)

        ready = False

        while not ready:

            success, frame = camera.read()

            if not success:
                continue

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                f"NEXT LETTER: {letter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                display_frame,
                f"Press {letter} to start",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Q = Quit",
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "SignSync Calibration",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                raise KeyboardInterrupt

            if key == ord(letter.lower()):
                ready = True


        for countdown in [3, 2, 1]:

            start_time = time.time()

            while time.time() - start_time < 1:

                success, frame = camera.read()

                if not success:
                    continue

                display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    f"GET READY: {letter}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    display_frame,
                    str(countdown),
                    (550, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 255),
                    8
                )

                cv2.imshow(
                    "SignSync Calibration",
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    raise KeyboardInterrupt


        collected = 0

        print(f"Collecting {letter}...")

        while collected < SAMPLES_PER_LETTER:

            success, frame = camera.read()

            if not success:
                continue

            features = extract_features(frame)

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                f"LETTER: {letter}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                display_frame,
                f"SAMPLES: {collected}/{SAMPLES_PER_LETTER}",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            if features is not None:

                writer.writerow(
                    [letter] + features.tolist()
                )

                csv_file.flush()

                collected += 1

                cv2.putText(
                    display_frame,
                    "CAPTURED",
                    (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                print(
                    f"\r{letter}: "
                    f"{collected}/{SAMPLES_PER_LETTER}",
                    end=""
                )

            else:

                cv2.putText(
                    display_frame,
                    "NO HAND DETECTED",
                    (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "SignSync Calibration",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                raise KeyboardInterrupt

            time.sleep(
                CAPTURE_DELAY
            )

        print()
        print(f"{letter} completed.")


finally:

    csv_file.close()

    camera.release()

    cv2.destroyAllWindows()

    hand_landmarker.close()


print()
print("=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)
print()
print(f"Dataset saved to:")
print(OUTPUT_FILE)
print()
print("Next step: train the live webcam classifier.")
print("=" * 60)