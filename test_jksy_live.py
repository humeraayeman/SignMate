import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ml" / "trained" / "signsync_prathum_jksy.keras"
LABELS_PATH = BASE_DIR / "ml" / "trained" / "prathum_jksy_labels.json"
LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"


model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)


def extract_features(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(image)

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


cap = cv2.VideoCapture(0)

print()
print("=" * 60)
print("J K S Y LIVE TEST")
print("=" * 60)
print("Show J, K, S or Y to the camera.")
print("Press Q to quit.")
print("=" * 60)
print()


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error.")
        break

    features = extract_features(frame)

    label = "---"
    confidence = 0.0

    if features is not None:

        prediction = model.predict(
            np.expand_dims(features, axis=0),
            verbose=0
        )[0]

        index = int(np.argmax(prediction))

        label = labels[index]
        confidence = float(prediction[index])

    cv2.putText(
        frame,
        f"{label}  {confidence * 100:.1f}%",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow(
        "SignMate - J K S Y Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()