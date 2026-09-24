import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from .ml_service import hand_landmarker


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "signsync_prathum_ck.keras"
)

LABELS_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "prathum_ck_labels.json"
)


ck_model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    ck_labels = json.load(f)


TARGET_SIGNS = {"C", "K"}


def extract_ck_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(image)

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


def predict_ck(frame):
    features = extract_ck_features(frame)

    if features is None:
        return None, 0.0

    prediction = ck_model.predict(
        np.expand_dims(features, axis=0),
        verbose=0
    )[0]

    index = int(np.argmax(prediction))

    return (
        ck_labels[index],
        float(prediction[index])
    )