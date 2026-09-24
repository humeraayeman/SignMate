import cv2
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# SignSync - Hand Detection
# ==========================================

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Hand Landmarker model
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


# ==========================================
# Check model file
# ==========================================

if not MODEL_PATH.exists():
    print()
    print("ERROR: Hand Landmarker model not found!")
    print()
    print("Expected location:")
    print(MODEL_PATH)
    print()
    print("Make sure this file exists:")
    print("models/hand_landmarker.task")
    print()
    exit()


print("Model found:")
print(MODEL_PATH)


# ==========================================
# Create MediaPipe Hand Landmarker
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================
# Start webcam
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print()
    print("ERROR: Could not open webcam.")
    print("Make sure your webcam is available.")
    print()

    detector.close()
    exit()


print()
print("==========================================")
print(" SignSync Hand Detection Started")
print("==========================================")
print()
print("Show your hand to the camera.")
print("Press Q to quit.")
print()


# Timestamp for MediaPipe video mode
frame_timestamp = 0


# ==========================================
# Main loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read webcam frame.")
        break


    # Mirror image
    frame = cv2.flip(frame, 1)


    # Get frame dimensions
    height, width, _ = frame.shape


    # ======================================
    # Convert OpenCV BGR → RGB
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # Create MediaPipe image
    # ======================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ======================================
    # Detect hand
    # ======================================

    frame_timestamp += 33

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # ======================================
    # Draw hand landmarks
    # ======================================

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            # --------------------------------
            # 21 landmark points
            # --------------------------------

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Keep points inside frame
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # --------------------------------
            # Hand connections
            # --------------------------------

            connections = [

                # Thumb
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                # Index finger
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                # Middle finger
                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                # Ring finger
                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                # Little finger
                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                # Palm
                (0, 17)
            ]


            # Draw connections
            for start, end in connections:

                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)

                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


        # ------------------------------------
        # Hand detected text
        # ------------------------------------

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    else:

        # ------------------------------------
        # No hand
        # ------------------------------------

        cv2.putText(
            frame,
            "SHOW YOUR HAND",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ======================================
    # Display webcam
    # ======================================

    cv2.imshow(
        "SignSync - Hand Detection",
        frame
    )


    # ======================================
    # Quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

camera.release()

detector.close()

cv2.destroyAllWindows()

print()
print("SignSync hand detection stopped.")