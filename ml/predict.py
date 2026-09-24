import json
from collections import Counter, deque
from pathlib import Path

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
    / "signsync_alphabet_normalized.keras"
)

LABEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "labels_normalized.json"
)

HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    MODEL_PATH,
    LABEL_PATH,
    HAND_MODEL_PATH
]:

    if not file_path.exists():

        print()
        print("ERROR: Required file not found:")
        print(file_path)
        print()

        exit()


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("==============================================")
print(" SignSync - Live Sign Recognition")
print("==============================================")
print()

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    labels = json.load(file)


print(
    "Model loaded successfully."
)

print(
    "Classes:",
    len(labels)
)

print()


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)


options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_hand_features(hand):

    if hand is None:

        return [0.0] * 63


    wrist = hand[0]

    raw_features = []


    # X
    for landmark in hand:

        raw_features.append(
            landmark.x - wrist.x
        )


    # Y
    for landmark in hand:

        raw_features.append(
            landmark.y - wrist.y
        )


    # Z
    for landmark in hand:

        raw_features.append(
            landmark.z - wrist.z
        )


    # --------------------------------------------------------
    # SIZE NORMALIZATION
    # --------------------------------------------------------

    xs = [
        landmark.x
        for landmark in hand
    ]

    ys = [
        landmark.y
        for landmark in hand
    ]


    width = max(xs) - min(xs)

    height = max(ys) - min(ys)

    size = max(
        width,
        height,
        0.001
    )


    normalized = [

        value / size

        for value in raw_features

    ]


    return normalized


# ============================================================
# GET HANDS
# ============================================================

def get_hands(result):

    if not result.hand_landmarks:

        return None, None


    hands = []


    for hand in result.hand_landmarks:

        features = extract_hand_features(
            hand
        )

        wrist_x = hand[0].x

        hands.append(
            (wrist_x, features)
        )


    # --------------------------------------------------------
    # ONE HAND
    # --------------------------------------------------------

    if len(hands) == 1:

        return (
            hands[0][1],
            [0.0] * 63
        )


    # --------------------------------------------------------
    # TWO HANDS
    # --------------------------------------------------------

    hands.sort(
        key=lambda item: item[0]
    )


    return (
        hands[0][1],
        hands[1][1]
    )


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    detector.close()

    exit()


# ============================================================
# SETTINGS
# ============================================================

frame_timestamp = 0

prediction_history = deque(
    maxlen=10
)

CONFIDENCE_THRESHOLD = 0.65


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Could not read webcam."
        )

        break


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    frame_timestamp += 33


    result = detector.detect_for_video(

        mp_image,

        frame_timestamp
    )


    # --------------------------------------------------------
    # HANDS
    # --------------------------------------------------------

    hand1, hand2 = get_hands(
        result
    )


    hand_count = 0

    if hand1 is not None:
        hand_count += 1

    if hand2 is not None:
        hand_count += 1


    # --------------------------------------------------------
    # DRAW LANDMARKS
    # --------------------------------------------------------

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                x = max(
                    0,
                    min(
                        x,
                        width - 1
                    )
                )

                y = max(
                    0,
                    min(
                        y,
                        height - 1
                    )
                )


                cv2.circle(

                    frame,

                    (x, y),

                    5,

                    (0, 255, 0),

                    -1
                )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if hand_count > 0:

        features = (
            hand1
            + hand2
        )


        input_data = np.array(
            [features],
            dtype=np.float32
        )


        predictions = model.predict(

            input_data,

            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(predictions)
        )


        confidence = float(
            predictions[predicted_index]
        )


        predicted_label = labels[
            predicted_index
        ]


        # ----------------------------------------------------
        # CONFIDENCE FILTER
        # ----------------------------------------------------

        if confidence >= CONFIDENCE_THRESHOLD:

            prediction_history.append(
                predicted_label
            )


        else:

            prediction_history.append(
                "?"
            )


        # ----------------------------------------------------
        # SMOOTHING
        # ----------------------------------------------------

        valid_predictions = [

            item

            for item in prediction_history

            if item != "?"

        ]


        if valid_predictions:

            stable_label = Counter(
                valid_predictions
            ).most_common(1)[0][0]

        else:

            stable_label = "?"


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Sign: {stable_label}",

            (20, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 255, 0),

            3
        )


        cv2.putText(

            frame,

            f"Confidence: {confidence * 100:.1f}%",

            (20, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            f"Hands: {hand_count}",

            (20, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


    else:

        prediction_history.clear()


        cv2.putText(

            frame,

            "SHOW YOUR HAND(S)",

            (20, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 0, 255),

            2
        )


    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    cv2.putText(

        frame,

        "Press Q to quit",

        (20, height - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(

        "SignSync - Live Recognition",

        frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

detector.close()

cv2.destroyAllWindows()

print()
print("SignSync recognition stopped.")