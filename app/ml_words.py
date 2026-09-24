"""Word model: 126 landmark features -> class label."""

from pathlib import Path
import json
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "app" / "models" / "signsync_words.keras"
LABELS_PATH = BASE_DIR / "app" / "models" / "labels_words.json"
NORM_PATH = BASE_DIR / "app" / "models" / "words_normalization.npz"

word_model = None
word_classes = []
norm_mean = None
norm_std = None

def init_word_model():
    global word_model, word_classes, norm_mean, norm_std

    if not MODEL_PATH.exists() or not LABELS_PATH.exists() or not NORM_PATH.exists():
        return False

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            word_classes = meta["classes"]

        norm_data = np.load(str(NORM_PATH))
        norm_mean = norm_data["mean"]
        norm_std = norm_data["std"]

        word_model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Loaded word classifier.")
        return True
    except Exception as e:
        print(f"Could not load word classifier ({e}).")
        return False


# Attempt eager initialization
init_word_model()


def predict_word_from_features(features_126: np.ndarray, min_confidence: float = 0.65):
    """Return (label, conf) or (None, 0) if below min_confidence."""
    global word_model, word_classes, norm_mean, norm_std

    if word_model is None:
        if not init_word_model():
            return None, 0.0

    try:
        norm_feat = (features_126 - norm_mean) / norm_std
        inp = np.expand_dims(norm_feat, axis=0)
        preds = word_model.predict(inp, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        conf = float(preds[best_idx])

        if conf >= min_confidence:
            return word_classes[best_idx].replace("_", " "), conf
        return None, conf
    except Exception as e:
        print("Word prediction error:", e)
        return None, 0.0
