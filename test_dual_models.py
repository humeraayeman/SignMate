from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# EXISTING REALSIGN MODEL
# ============================================================

OLD_MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "signsync_realsign.keras"
)

OLD_LABELS_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "labels_realsign.json"
)

OLD_NORMALIZATION_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "realsign_normalization.npz"
)


# ============================================================
# NEW PRATHUM MODEL
# ============================================================

NEW_MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "signsync_prathum_targets_r.keras"
)

NEW_LABELS_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "prathum_targets_r_labels.json"
)


# ============================================================
# MEDIAPIPE
# ============================================================

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# LOAD OLD MODEL
# ============================================================

print()
print("=" * 60)
print("Loading existing RealSign model...")
print("=" * 60)

old_model = tf.keras.models.load_model(
    OLD_MODEL_PATH
)

with open(
    OLD_LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:
    old_labels = json.load(f)

old_normalization = np.load(
    OLD_NORMALIZATION_PATH
)

old_mean = old_normalization["mean"].astype(
    np.float32
)

old_std = old_normalization["std"].astype(
    np.float32
)

old_std[old_std < 1e-6] = 1.0


# ============================================================
# LOAD NEW MODEL
# ============================================================

print()
print("=" * 60)
print("Loading Prathum target model...")
print("=" * 60)

new_model = tf.keras.models.load_model(
    NEW_MODEL_PATH
)

with open(
    NEW_LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:
    new_labels = json.load(f)


# ============================================================
# NORMALIZATION FOR OLD MODEL
# ============================================================

def normalize_old_hand(landmarks):

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
# NORMALIZATION FOR NEW PRATHUM MODEL
# ============================================================

def normalize_new_hand(landmarks):

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    wrist = points[0].copy()

    points = points - wrist

    scale = np.max(
        np.linalg.norm(
            points[:, :2],
            axis=1
        )
    )

    if scale < 1e-6:
        scale = 1.0

    points = points / scale

    return points.flatten()


# ============================================================
# EXTRACT BOTH FEATURE TYPES
# ============================================================

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
        return None, 0

    hands = list(
        result.hand_landmarks
    )

    hands.sort(
        key=lambda h: h[0].x
    )

    old_features = []
    new_features = []

    for hand in hands[:2]:

        old_features.extend(
            normalize_old_hand(hand)
        )

        new_features.extend(
            normalize_new_hand(hand)
        )

    while len(old_features) < 126:
        old_features.extend(
            [0.0] * 63
        )

    while len(new_features) < 126:
        new_features.extend(
            [0.0] * 63
        )

    return (
        np.array(
            old_features[:126],
            dtype=np.float32
        ),
        np.array(
            new_features[:126],
            dtype=np.float32
        ),
        len(hands)
    )


# ============================================================
# PREDICT OLD MODEL
# ============================================================

def predict_old(features):

    normalized = (
        features - old_mean
    ) / old_std

    prediction = old_model.predict(
        normalized.reshape(1, -1),
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    return (
        old_labels[index],
        float(prediction[index])
    )


# ============================================================
# PREDICT NEW MODEL
# ============================================================

def predict_new(features):

    prediction = new_model.predict(
        features.reshape(1, -1),
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    return (
        new_labels[index],
        float(prediction[index])
    )


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

options = mp.tasks.vision.HandLandmarkerOptions(

    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH
        )
    ),

    running_mode=(
        mp.tasks.vision.RunningMode.IMAGE
    ),

    num_hands=2,

    min_hand_detection_confidence=0.40,

    min_hand_presence_confidence=0.40,

    min_tracking_confidence=0.40
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    raise SystemExit


print()
print("=" * 60)
print("DUAL MODEL TEST")
print("=" * 60)
print()
print("Test:")
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
            break

        display = frame.copy()

        result = extract_features(
            frame,
            detector
        )

        if result[0] is not None:

            old_features = result[0]
            new_features = result[1]
            hand_count = result[2]

            old_label, old_confidence = (
                predict_old(
                    old_features
                )
            )

            new_label, new_confidence = (
                predict_new(
                    new_features
                )
            )

            # ------------------------------------------------
            # DISPLAY OLD MODEL
            # ------------------------------------------------

            cv2.putText(
                display,
                f"OLD: {old_label} "
                f"{old_confidence * 100:.1f}%",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # DISPLAY NEW MODEL
            # ------------------------------------------------

            cv2.putText(
                display,
                f"NEW: {new_label} "
                f"{new_confidence * 100:.1f}%",
                (25, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Hands: {hand_count}",
                (25, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display,
                "No hand detected",
                (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "SignSync - Old vs New Model",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

