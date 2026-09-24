from pathlib import Path
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ml" / "trained" / "signsync_prathum_targets.keras"
LABELS_PATH = BASE_DIR / "ml" / "trained" / "prathum_targets_labels.json"
LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"


with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)


def normalize_hand(hand):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    wrist = points[0].copy()
    points -= wrist

    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale < 1e-6:
        scale = 1.0

    points /= scale

    return points.flatten()


def extract_features(frame, detector):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(image)

    if not result.hand_landmarks:
        return None

    hands = list(result.hand_landmarks)

    hands.sort(key=lambda h: h[0].x)

    features = []

    for hand in hands[:2]:
        features.extend(
            normalize_hand(hand)
        )

    while len(features) < 126:
        features.extend([0.0] * 63)

    return np.array(
        features[:126],
        dtype=np.float32
    )


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit


print()
print("=" * 60)
print("Prathum Target Model - Webcam Test")
print("=" * 60)
print()
print("Target signs:")
print("I  O  R  T  V  Y")
print()
print("Press Q to quit.")
print()


with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as detector:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        features = extract_features(
            frame,
            detector
        )

        display = frame.copy()

        if features is not None:

            probabilities = model.predict(
                features.reshape(1, -1),
                verbose=0
            )[0]

            index = int(
                np.argmax(probabilities)
            )

            prediction = labels[index]
            confidence = float(
                probabilities[index]
            )

            text = (
                f"{prediction} "
                f"{confidence * 100:.1f}%"
            )

            cv2.putText(
                display,
                text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

        else:

            cv2.putText(
                display,
                "No hand detected",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "SignSync - Prathum Target Test",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()