from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "app" / "ml_service.py"

text = FILE.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. Add correction paths
# ------------------------------------------------------------

marker = '''HAND_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)
'''

addition = marker + '''

CORRECTION_MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "signsync_prathum_correction.keras"
)

CORRECTION_LABELS_PATH = (
    BASE_DIR
    / "ml"
    / "trained"
    / "prathum_correction_labels.json"
)
'''

if "CORRECTION_MODEL_PATH" not in text:
    text = text.replace(marker, addition)


# ------------------------------------------------------------
# 2. Load correction model
# ------------------------------------------------------------

marker = '''model = tf.keras.models.load_model(
    MODEL_PATH
)
'''

addition = marker + '''

print("Loading Prathum correction model...")

if not CORRECTION_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Correction model not found:\\n{CORRECTION_MODEL_PATH}"
    )

if not CORRECTION_LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Correction labels not found:\\n{CORRECTION_LABELS_PATH}"
    )

correction_model = tf.keras.models.load_model(
    CORRECTION_MODEL_PATH
)

with open(
    CORRECTION_LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:
    correction_labels = json.load(f)

TARGET_SIGNS = {
    "I",
    "O",
    "R",
    "S",
    "T",
    "V",
    "Y"
}

print("Correction model loaded.")
print("Correction classes:", correction_labels)
'''

if "correction_model = tf.keras.models.load_model" not in text:
    text = text.replace(marker, addition)


# ------------------------------------------------------------
# 3. Add correction feature extraction
# ------------------------------------------------------------

marker = '''# ============================================================
# MODEL PREDICTION
# ============================================================
'''

addition = '''# ============================================================
# PRATHUM CORRECTION FEATURE EXTRACTION
# ============================================================

def extract_correction_features(frame):

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

        wrist = hand[0]

        points = []

        for landmark in hand:

            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z

            points.append([
                x,
                y,
                z
            ])

        points = np.asarray(
            points,
            dtype=np.float32
        )

        radial_distance = np.sqrt(
            points[:, 0] ** 2 +
            points[:, 1] ** 2
        )

        scale = float(
            np.max(radial_distance)
        )

        if scale < 1e-6:
            scale = 1.0

        points = points / scale

        hands.append({
            "x": wrist.x,
            "features": points.flatten()
        })

    hands.sort(
        key=lambda h: h["x"]
    )

    first = hands[0]["features"]

    if len(hands) == 1:

        second = np.zeros(
            63,
            dtype=np.float32
        )

    else:

        second = hands[1]["features"]

    return np.concatenate([
        first,
        second
    ]).astype(np.float32)


def predict_correction(frame):

    features = extract_correction_features(
        frame
    )

    if features is None:
        return None, 0.0

    prediction = correction_model.predict(
        np.expand_dims(features, axis=0),
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[index]
    )

    label = correction_labels[index]

    return label, confidence


# ============================================================
# MODEL PREDICTION
# ============================================================
'''

if "def extract_correction_features" not in text:
    text = text.replace(marker, addition)


# ------------------------------------------------------------
# 4. Replace predict_frame
# ------------------------------------------------------------

start = text.index("def predict_frame(frame):")
end = text.index(
    "# ============================================================\n# READY",
    start
)

new_predict_frame = '''def predict_frame(frame):

    # --------------------------------------------------------
    # OLD REALSIGN MODEL
    # --------------------------------------------------------

    original = predict_single(frame)

    flipped_frame = cv2.flip(
        frame,
        1
    )

    flipped = predict_single(
        flipped_frame
    )

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

    if original["hands"] == 0:
        best = flipped

    elif flipped["hands"] == 0:
        best = original

    elif flipped["confidence"] > original["confidence"]:
        best = flipped

    else:
        best = original

    label = best["label"]
    confidence = best["confidence"]
    hands = best["hands"]

    # --------------------------------------------------------
    # NEW PRATHUM CORRECTION MODEL
    #
    # Only target the seven problematic signs.
    # All other letters remain handled by RealSign.
    # --------------------------------------------------------

    correction_label, correction_confidence = (
        predict_correction(frame)
    )

    if (
        correction_label in TARGET_SIGNS
        and
        correction_confidence >= 0.50
    ):

        return {
            "success": True,
            "label": correction_label,
            "confidence": correction_confidence,
            "hands": hands,
            "message": "Correction model prediction."
        }

    # --------------------------------------------------------
    # ORIGINAL REALSIGN RESULT
    # --------------------------------------------------------

    if confidence < 0.35:

        return {
            "success": False,
            "label": label,
            "confidence": confidence,
            "hands": hands,
            "message": "Low confidence."
        }

    return {
        "success": True,
        "label": label,
        "confidence": confidence,
        "hands": hands,
        "message": "Prediction successful."
    }


'''

text = text[:start] + new_predict_frame + text[end:]

FILE.write_text(text, encoding="utf-8")

print("=" * 60)
print("CORRECTION MODEL INTEGRATED")
print("=" * 60)
print(FILE)
print()
print("Target signs:")
print("I O R S T V Y")
print()
print("Original RealSign model remains active for all other letters.")
print("=" * 60)