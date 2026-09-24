"""
SignMate - Indian Sign Language Word Webcam Image Collector
===========================================================
Interactive tool to capture real-time webcam photos into the
image dataset for words, mirroring the letter image collection pipeline.

Usage:
    python collect_word_images.py --word NAMASTE --count 20
    python collect_word_images.py --list-words

Outputs:
    dataset/words_dataset/<WORD>/<index>.jpg
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "words_dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

WORDS_META = BASE_DIR / "dataset" / "words" / "words_metadata.json"


def load_standard_words():
    if WORDS_META.exists():
        with open(WORDS_META, "r", encoding="utf-8") as f:
            return [w["word"] for w in json.load(f).get("words", [])]
    return [
        "NAMASTE", "HELLO", "GOOD MORNING", "GOOD NIGHT", "WELCOME", "HOW ARE YOU",
        "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "OKAY", "GOOD", "BAD",
        "HELP", "STOP", "PAIN", "HOSPITAL", "DOCTOR", "MOTHER", "FATHER",
        "FRIEND", "FAMILY", "TEACHER", "SCHOOL", "BOOK", "HOME", "WORK",
        "MONEY", "EAT", "WATER", "SLEEP", "HAPPY", "SAD", "I LOVE YOU",
        "WHAT", "WHERE", "WHY", "WHEN", "WHO"
    ]


def collect_images_for_word(word: str, count: int = 20, delay_sec: float = 0.4):
    slug = word.strip().upper().replace(" ", "_")
    target_dir = DATASET_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find highest existing index
    existing_files = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png"))
    start_index = len(existing_files)

    print(f"\n{'='*65}")
    print(f" SignMate Word Image Collector: {word.upper()} ({slug})")
    print(f" Target: {count} images (Starting index: {start_index})")
    print(f" Folder: {target_dir}")
    print(f"{'='*65}\n")
    print("Press 'q' in the camera window to quit, or 's' to manually snap.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Countdown phase
    countdown_start = time.time()
    while time.time() - countdown_start < 3.0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remaining = int(np.ceil(3.0 - (time.time() - countdown_start)))

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.putText(frame, f"Word: {word.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"GET READY IN: {remaining}s", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.imshow("SignMate Word Image Collector", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    captured = 0
    last_capture_time = time.time()

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        current_time = time.time()
        ready_to_snap = (current_time - last_capture_time) >= delay_sec

        h, w, _ = frame.shape
        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 120, 0) if ready_to_snap else (50, 50, 50), -1)
        cv2.putText(frame, f"CAPTURING: {word.upper()} ({captured + 1}/{count})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Hold gesture steady in frame...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Central target reticle
        box_w, box_h = 360, 300
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2 + 30
        cv2.rectangle(frame, (bx1, by1), (bx1 + box_w, by1 + box_h), (0, 255, 120), 2)

        cv2.imshow("SignMate Word Image Collector", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            print("\nCollection aborted by user.")
            break

        if ready_to_snap or key == ord('s'):
            img_filename = f"{start_index + captured}.jpg"
            img_path = target_dir / img_filename
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  [OK] Saved image {img_filename} to {slug}")
            captured += 1
            last_capture_time = current_time

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Captured {captured} images for {word.upper()} in {target_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="SignMate ISL Word Image Dataset Collector")
    parser.add_argument("--word", type=str, help="Word to collect images for (e.g. NAMASTE, HELLO, GOOD)")
    parser.add_argument("--count", type=int, default=20, help="Number of images to capture (default: 20)")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between auto-captured frames (default: 0.4)")
    parser.add_argument("--list-words", action="store_true", help="List standard 40 ISL vocabulary words")

    args = parser.parse_args()
    words = load_standard_words()

    if args.list_words:
        print("\nSignMate Standard Word Vocabulary:")
        print("-----------------------------------")
        for idx, w in enumerate(words, 1):
            print(f" {idx:2d}. {w}")
        print()
        return

    if not args.word:
        print("Error: Please provide --word <NAME> or --list-words.")
        print("Example: python collect_word_images.py --word NAMASTE --count 25")
        sys.exit(1)

    collect_images_for_word(args.word, args.count, args.delay)


if __name__ == "__main__":
    main()
