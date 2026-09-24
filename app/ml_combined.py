import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from .ml_service import predict_frame as predict_realsign, hand_landmarker
from .ml_ck import predict_ck
from .ml_jksy import predict_jksy

logger = logging.getLogger("signmate.ml")
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Prathum correction model
CORRECTION_MODEL_PATH = BASE_DIR / "ml" / "trained" / "signsync_prathum_correction.keras"
CORRECTION_LABELS_PATH = BASE_DIR / "ml" / "trained" / "prathum_correction_labels.json"

correction_model = tf.keras.models.load_model(CORRECTION_MODEL_PATH)
with open(CORRECTION_LABELS_PATH, "r", encoding="utf-8") as f:
    correction_labels = json.load(f)

TARGET_SIGNS = {"I", "O", "R", "S", "T", "V", "Y"}


def extract_correction_features(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extracts normalized dual-hand landmarks for the correction model."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(image)

    if not result.hand_landmarks:
        return None

    hands = []
    for hand in result.hand_landmarks:
        wrist = hand[0]
        points = np.array(
            [[p.x - wrist.x, p.y - wrist.y, p.z - wrist.z] for p in hand],
            dtype=np.float32
        )
        radial = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        scale = float(np.max(radial))
        if scale < 1e-6:
            scale = 1.0
        points /= scale
        hands.append((wrist.x, points.flatten()))

    hands.sort(key=lambda x: x[0])
    first = hands[0][1]
    second = hands[1][1] if len(hands) > 1 else np.zeros(63, dtype=np.float32)

    return np.concatenate([first, second])


def predict_correction(frame: np.ndarray) -> Tuple[Optional[str], float]:
    """Runs inference with the secondary correction model."""
    features = extract_correction_features(frame)
    if features is None:
        return None, 0.0

    prediction = correction_model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
    index = int(np.argmax(prediction))
    return correction_labels[index], float(prediction[index])


def predict_frame(frame: np.ndarray) -> Dict[str, Any]:
    """
    Arbitrates multi-model predictions across RealSign, C/K, J/K/S/Y, and correction models.
    """
    # 1. Primary RealSign 26-Letter Model
    original = predict_realsign(frame)
    original_label = original.get("label")
    original_confidence = float(original.get("confidence", 0.0))
    hands_detected = original.get("hands", 1)

    # 2. High confidence preservation for standard letters
    if (
        original.get("success")
        and original_confidence >= 0.70
        and original_label not in {"C", "K", "J", "S", "Y"}
    ):
        return {**original, "is_word": False}

    # 3. C / K Specialist Classifier
    if original_label in {"C", "K"} or original_confidence < 0.60:
        ck_label, ck_confidence = predict_ck(frame)
        if (
            ck_label in {"C", "K"}
            and ck_confidence >= 0.80
            and ck_confidence > original_confidence
        ):
            return {
                "success": True,
                "label": ck_label,
                "is_word": False,
                "confidence": ck_confidence,
                "hands": hands_detected,
                "message": "C/K specialist prediction."
            }

    # 4. J / K / S / Y Specialist Classifier
    if original_label in {"J", "K", "S", "Y"} or original_confidence < 0.60:
        jksy_label, jksy_confidence = predict_jksy(frame)
        if (
            jksy_label in {"J", "K", "S", "Y"}
            and jksy_confidence >= 0.72
            and jksy_confidence > original_confidence
        ):
            return {
                "success": True,
                "label": jksy_label,
                "is_word": False,
                "confidence": jksy_confidence,
                "hands": hands_detected,
                "message": "J/K/S/Y specialist prediction."
            }

    # 5. Prathum Target Correction (I, O, R, S, T, V, Y)
    if original_label in TARGET_SIGNS or original_confidence < 0.55:
        correction_label, correction_confidence = predict_correction(frame)
        if (
            correction_label in TARGET_SIGNS
            and correction_label != "OTHER"
            and correction_confidence >= 0.78
            and correction_confidence > original_confidence
        ):
            return {
                "success": True,
                "label": correction_label,
                "is_word": False,
                "confidence": correction_confidence,
                "hands": hands_detected,
                "message": "Target correction prediction."
            }

    # 6. Fallback to Primary Model
    return {**original, "is_word": False}