from pathlib import Path
import json

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "signmate_realsign.keras"
)

LABELS_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "labels_realsign.json"
)

NORMALIZATION_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "realsign_normalization.npz"
)

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 60)
print("SIGN SYNC - LOADING REALSIGN ML MODEL")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Labels file not found:\n{LABELS_PATH}"
    )

if not NORMALIZATION_PATH.exists():
    raise FileNotFoundError(
        f"Normalization file not found:\n{NORMALIZATION_PATH}"
    )

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{HAND_MODEL_PATH}"
    )


# ============================================================
# LOAD TENSORFLOW MODEL
# ============================================================

print("Loading TensorFlow model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:

    labels = json.load(f)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

normalization = np.load(
    NORMALIZATION_PATH
)

feature_mean = normalization["mean"].astype(
    np.float32
)

feature_std = normalization["std"].astype(
    np.float32
)

feature_std[
    feature_std < 1e-6
] = 1.0


print("Model loaded successfully.")
print("Classes:", labels)
print("Number of classes:", len(labels))


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(
        HAND_MODEL_PATH
    )
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,

    running_mode=vision.RunningMode.IMAGE,

    num_hands=2,

    min_hand_detection_confidence=0.40,

    min_hand_presence_confidence=0.40,

    min_tracking_confidence=0.40
)

hand_landmarker = (
    vision.HandLandmarker
    .create_from_options(options)
)


# ============================================================
# NORMALIZE ONE HAND
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

    min_x = np.min(
        points[:, 0]
    )

    max_x = np.max(
        points[:, 0]
    )

    min_y = np.min(
        points[:, 1]
    )

    max_y = np.max(
        points[:, 1]
    )

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
# EXTRACT 126 FEATURES
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

        return None, 0


    hands = []


    for hand in detected_hands:

        features = normalize_hand(
            hand
        )

        wrist_x = hand[0].x

        hands.append({
            "x": wrist_x,
            "features": features
        })


    # Sort hands consistently.
    hands.sort(
        key=lambda h: h["x"]
    )


    # ========================================================
    # ONE HAND
    # ========================================================

    if len(hands) == 1:

        first = hands[0]["features"]

        second = np.zeros(
            63,
            dtype=np.float32
        )

        features = np.concatenate([
            first,
            second
        ])

        return features, 1


    # ========================================================
    # TWO HANDS
    # ========================================================

    first = hands[0]["features"]

    second = hands[1]["features"]

    features = np.concatenate([
        first,
        second
    ])

    return features, 2


# ============================================================
# NORMALIZE FEATURES USING TRAINING STATISTICS
# ============================================================

def normalize_features(features):

    features = np.asarray(
        features,
        dtype=np.float32
    )

    normalized = (
        features - feature_mean
    ) / feature_std

    return normalized.astype(
        np.float32
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_features(features):

    normalized_features = (
        normalize_features(
            features
        )
    )

    input_data = np.expand_dims(
        normalized_features,
        axis=0
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_index]
    )

    label = labels[class_index]

    return (
        label,
        confidence,
        prediction
    )


# ============================================================
# SINGLE FRAME PREDICTION
# ============================================================

def predict_single(frame):

    features, hand_count = (
        extract_features(frame)
    )

    if features is None:

        return {
            "success": False,
            "label": None,
            "confidence": 0.0,
            "hands": 0,
            "prediction": None,
            "message": "No hand detected."
        }


    label, confidence, prediction = (
        predict_features(
            features
        )
    )


    return {
        "success": True,
        "label": label,
        "confidence": confidence,
        "hands": hand_count,
        "prediction": prediction.tolist(),
        "message": "Prediction successful."
    }


# ============================================================
# FINAL FRAME PREDICTION
# ============================================================

def predict_frame(frame):

    # --------------------------------------------------------
    # ORIGINAL FRAME
    # --------------------------------------------------------

    original = predict_single(
        frame
    )


    # --------------------------------------------------------
    # MIRRORED FRAME
    # --------------------------------------------------------

    flipped_frame = cv2.flip(
        frame,
        1
    )

    flipped = predict_single(
        flipped_frame
    )


    # --------------------------------------------------------
    # NO HAND
    # --------------------------------------------------------

    if (
        not original["success"]
        and
        not flipped["success"]
    ):

        return {
            "success": False,
            "label": None,
            "confidence": 0.0,
            "hands": 0,
            "message": "No hand detected."
        }


    # --------------------------------------------------------
    # SELECT PREDICTION
    # --------------------------------------------------------

    if original["hands"] == 0:

        best = flipped

    elif flipped["hands"] == 0:

        best = original

    else:

        if (
            flipped["confidence"]
            >
            original["confidence"]
        ):

            best = flipped

        else:

            best = original


    label = best["label"]

    confidence = best["confidence"]

    hands = best["hands"]


    # --------------------------------------------------------
    # CONFIDENCE THRESHOLD
    # --------------------------------------------------------

    if confidence < 0.35:

        return {
            "success": False,
            "label": label,
            "confidence": confidence,
            "hands": hands,
            "message": "Low confidence."
        }


    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return {
        "success": True,
        "label": label,
        "confidence": confidence,
        "hands": hands,
        "message": "Prediction successful."
    }


# ============================================================
# READY
# ============================================================

print()
print("=" * 60)
print("REALSign ML SERVICE READY")
print("=" * 60)
print("Model:", MODEL_PATH.name)
print("Classes:", len(labels))
print("Confidence threshold: 35%")
print("=" * 60)
print()