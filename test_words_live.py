"""
SignMate - Live Webcam Word Sign Prediction Tester
==================================================
Runs real-time word sign recognition from your webcam using
the neural network model trained on the word image dataset.

Usage:
    python test_words_live.py
"""

from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "app" / "models" / "signsync_words.keras"
LABELS_PATH = BASE_DIR / "app" / "models" / "labels_words.json"
NORM_PATH = BASE_DIR / "app" / "models" / "words_normalization.npz"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

print("\n" + "=" * 65)
print(" SIGNMATE - LIVE WEBCAM WORD SIGN TESTER")
print("=" * 65 + "\n")

if not MODEL_PATH.exists() or not LABELS_PATH.exists() or not NORM_PATH.exists():
    print("Error: Trained word model not found.")
    print("Please run 'python train_words_model.py' first.\n")
    exit(1)

# Load labels & normalization
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
    classes = meta["classes"]

norm_data = np.load(str(NORM_PATH))
mean = norm_data["mean"]
std = norm_data["std"]

print(f"Loaded {len(classes)} word classes.")
print(f"Loading trained neural network from {MODEL_PATH}...")
model = tf.keras.models.load_model(str(MODEL_PATH))

print(f"Loading MediaPipe hand landmarker from {HAND_MODEL_PATH}...")
base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)
print("All models initialized successfully!\n")


def normalize_hand(landmarks):
    wrist = landmarks[0]
    points = []
    for landmark in landmarks:
        points.append([landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z])
    points = np.array(points, dtype=np.float32)
    width = np.max(points[:, 0]) - np.min(points[:, 0])
    height = np.max(points[:, 1]) - np.min(points[:, 1])
    scale = max(width, height)
    if scale < 1e-6:
        scale = 1.0
    return (points / scale).flatten()


def extract_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)

    detected_hands = result.hand_landmarks
    if not detected_hands:
        return None

    hands = []
    for hand in detected_hands:
        feats = normalize_hand(hand)
        hands.append({"x": hand[0].x, "features": feats})

    hands.sort(key=lambda h: h["x"])

    if len(hands) == 1:
        first = hands[0]["features"]
        second = np.zeros(63, dtype=np.float32)
        return np.concatenate([first, second])

    first = hands[0]["features"]
    second = hands[1]["features"]
    return np.concatenate([first, second])


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Live webcam window active. Show ISL word signs to camera.")
    print("Press 'q' in camera window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        features = extract_features(frame)
        pred_label = "No hands detected"
        pred_conf = 0.0

        if features is not None:
            norm_feat = (features - mean) / std
            input_arr = np.expand_dims(norm_feat, axis=0)
            preds = model.predict(input_arr, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            pred_conf = float(preds[best_idx])
            pred_label = classes[best_idx].replace("_", " ")

        # Draw UI overlay
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        if features is not None:
            color = (0, 255, 120) if pred_conf >= 0.60 else (0, 180, 255)
            cv2.putText(frame, f"WORD: {pred_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {pred_conf*100:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Show hands to camera...", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

        cv2.imshow("SignMate - Word Sign Live Tester", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
