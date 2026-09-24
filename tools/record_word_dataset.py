"""
Record MediaPipe hand landmarks for an ISL word.

    python tools/record_word_dataset.py --word NAMASTE --samples 20 --frames 30
    python tools/record_word_dataset.py --list-words

Writes npy + json under dataset/words/landmarks/<WORD>/.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("Error: MediaPipe is required. Run: pip install mediapipe")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = BASE_DIR / "dataset" / "words" / "landmarks"
METADATA_PATH = BASE_DIR / "dataset" / "words" / "words_metadata.json"


def load_standard_words():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [w["word"] for w in data.get("words", [])]
    return [
        "NAMASTE", "HELLO", "GOOD MORNING", "GOOD NIGHT", "WELCOME", "HOW ARE YOU",
        "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "OKAY", "GOOD", "BAD",
        "HELP", "STOP", "PAIN", "HOSPITAL", "DOCTOR", "MOTHER", "FATHER",
        "FRIEND", "FAMILY", "TEACHER", "SCHOOL", "BOOK", "HOME", "WORK",
        "MONEY", "EAT", "WATER", "SLEEP", "HAPPY", "SAD", "I LOVE YOU",
        "WHAT", "WHERE", "WHY", "WHEN", "WHO"
    ]


def extract_keypoints(results):
    """
    Extracts 126 features:
    63 for Left Hand (21 landmarks x 3) + 63 for Right Hand (21 landmarks x 3).
    If a hand is missing, returns zeros for that hand.
    """
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32).flatten()
            if label == "Left":
                lh = pts
            elif label == "Right":
                rh = pts

    return np.concatenate([lh, rh])


def record_word_samples(word: str, num_samples: int = 20, frames_per_sample: int = 30, output_dir: Path = DEFAULT_OUT_DIR):
    slug = word.strip().upper().replace(" ", "_")
    target_dir = output_dir / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Recording: {word.upper()} ({slug})")
    print(f" Targets: {num_samples} samples x {frames_per_sample} frames each")
    print(f" Directory: {target_dir}")
    print(f"{'='*60}\n")
    print("Press 'q' at any time in the camera window to abort.\n")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam. Please verify camera connection.")
        return

    # Determine starting sample index
    existing = list(target_dir.glob("sample_*.npy"))
    start_idx = len(existing) + 1

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for sample_idx in range(start_idx, start_idx + num_samples):
            # 1. Countdown preparation phase (3 seconds)
            countdown_start = time.time()
            while time.time() - countdown_start < 3.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                remaining = int(np.ceil(3.0 - (time.time() - countdown_start)))

                # Draw countdown banner
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
                cv2.putText(frame, f"Word: {word.upper()} | Sample {sample_idx}/{start_idx + num_samples - 1}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"GET READY IN: {remaining}s",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                cv2.imshow("SignMate Dataset Recorder", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("\nRecording canceled by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # 2. Recording frames
            sequence = []
            print(f"  [● RECORDING] Sample {sample_idx} ...", end="", flush=True)

            for frame_num in range(frames_per_sample):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for lms in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                # Draw recording UI
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 180), -1)
                cv2.putText(frame, f"● RECORDING: {word.upper()} (Frame {frame_num + 1}/{frames_per_sample})",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("SignMate Dataset Recorder", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("\nRecording aborted.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Save sample
            seq_arr = np.array(sequence, dtype=np.float32)  # shape: (frames, 126)
            out_base = target_dir / f"sample_{sample_idx:03d}"
            np.save(str(out_base) + ".npy", seq_arr)

            # Also save clean JSON for inspection
            with open(str(out_base) + ".json", "w", encoding="utf-8") as jf:
                json.dump({
                    "word": word.upper(),
                    "sample_id": sample_idx,
                    "frames": frames_per_sample,
                    "features_per_frame": 126,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }, jf, indent=2)

            print(f" Saved! ({seq_arr.shape})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSuccessfully finished recording {num_samples} samples for '{word.upper()}'!")
    print(f"Saved into: {target_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="SignMate ISL Word Landmark Recorder")
    parser.add_argument("--word", type=str, help="Word or phrase to record (e.g., NAMASTE, HELLO)")
    parser.add_argument("--samples", type=int, default=15, help="Number of repetitions to record (default: 15)")
    parser.add_argument("--frames", type=int, default=30, help="Frames per repetition (default: 30)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder")
    parser.add_argument("--list-words", action="store_true", help="List standard 40 ISL vocabulary words")

    args = parser.parse_args()
    words = load_standard_words()

    if args.list_words:
        print("\nSignMate 40 Standard ISL Vocabulary Words:")
        print("------------------------------------------")
        for i, w in enumerate(words, 1):
            print(f" {i:2d}. {w}")
        print()
        return

    if not args.word:
        print("Error: Please provide --word <NAME> or --list-words.")
        print("Example: python tools/record_word_dataset.py --word NAMASTE --samples 15")
        sys.exit(1)

    record_word_samples(
        word=args.word,
        num_samples=args.samples,
        frames_per_sample=args.frames,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
