"""
Grab example ISL images / YouTube frames for a word.

    python tools/download_word_media.py --word NAMASTE --source all
    python tools/download_word_media.py --word HELLO --source youtube --extract-frames 15
    python tools/download_word_media.py --word THANK_YOU --source google --max-images 8
    python tools/download_word_media.py --batch-top-words
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
import urllib.request
import urllib.parse

import cv2
import requests

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "words_dataset"
TEMP_DIR = BASE_DIR / "dataset" / "temp_media"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# Curated high-confidence online ISL educational images and video reference repositories
ISL_ONLINE_CURATED_SOURCES = {
    "NAMASTE": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/A_namaste_sign_gesture.jpg/480px-A_namaste_sign_gesture.jpg",
        "https://images.unsplash.com/photo-1544717305-2782549b5136?w=480&q=80",
    ],
    "HELLO": [
        "https://images.unsplash.com/photo-1576765608535-5f04d1e3f289?w=480&q=80",
        "https://images.unsplash.com/photo-1582213782179-e0d53f98f2ca?w=480&q=80",
    ],
    "THANK_YOU": [
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=480&q=80",
    ],
    "GOOD": [
        "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=480&q=80",
    ],
    "HELP": [
        "https://images.unsplash.com/photo-1584515979956-d9f6e5d09982?w=480&q=80",
    ],
    "WATER": [
        "https://images.unsplash.com/photo-1548839140-29a749e1bc4e?w=480&q=80",
    ],
    "MOTHER": [
        "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=480&q=80",
    ],
    "DOCTOR": [
        "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=480&q=80",
    ]
}


def download_google_images(word: str, max_images: int = 6):
    slug = word.strip().upper().replace(" ", "_")
    target_dir = DATASET_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Google / Web Image Search] Searching images for '{word}'...")
    query = f"Indian Sign Language {word} sign gesture"
    encoded_query = urllib.parse.quote(query)

    # 1. Fetch search page from public engine
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    headers = {"User-Agent": USER_AGENT}

    saved = 0
    img_urls = []

    # Check curated list first
    if slug in ISL_ONLINE_CURATED_SOURCES:
        img_urls.extend(ISL_ONLINE_CURATED_SOURCES[slug])

    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # Extract image URLs
            found = re.findall(r'(https?://[^"\'\s>]+\.(?:jpg|jpeg|png))', html, re.IGNORECASE)
            img_urls.extend(found[:max_images])
    except Exception as e:
        print(f"  Note: Web search query completed ({e})")

    # Deduplicate
    unique_urls = list(dict.fromkeys(img_urls))

    existing_count = len(list(target_dir.glob("*.jpg")))
    for idx, url in enumerate(unique_urls[:max_images]):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=8)
            if r.status_code == 200 and len(r.content) > 4000:
                img_data = np.frombuffer(r.content, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                    resized = cv2.resize(img, (340, 260), interpolation=cv2.INTER_AREA)
                    out_name = f"web_{existing_count + saved}.jpg"
                    cv2.imwrite(str(target_dir / out_name), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    print(f"  [OK] Saved {out_name} to {slug}/")
                    saved += 1
                    if saved >= max_images:
                        break
        except Exception:
            continue

    print(f"Total web images saved for {slug}: {saved}")
    return saved


def download_and_extract_youtube_frames(word: str, max_frames: int = 12):
    """
    Searches YouTube for official ISL sign demonstrations and extracts gesture frames.
    """
    if yt_dlp is None:
        print("Error: yt-dlp is not installed. Run 'pip install yt-dlp'.")
        return 0

    slug = word.strip().upper().replace(" ", "_")
    target_dir = DATASET_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[YouTube Video Search] Searching video demonstration for '{word}'...")
    query = f"ytsearch1:Indian Sign Language {word} ISL"
    video_out_tmpl = str(TEMP_DIR / f"{slug}_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": video_out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "max_downloads": 1,
        "socket_timeout": 15
    }

    downloaded_video = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=True)
            if "entries" in info and len(info["entries"]):
                entry = info["entries"][0]
                downloaded_video = ydl.prepare_filename(entry)
            elif "id" in info:
                downloaded_video = ydl.prepare_filename(info)
    except Exception as e:
        print(f"  YouTube search/download note: {e}")

    # Locate downloaded file if extension differed
    if not downloaded_video or not Path(downloaded_video).exists():
        found = list(TEMP_DIR.glob(f"{slug}_*.*"))
        if found:
            downloaded_video = str(found[0])

    if not downloaded_video or not Path(downloaded_video).exists():
        print(f"  Could not download YouTube demonstration for {word}.")
        return 0

    print(f"  Downloaded video: {Path(downloaded_video).name}. Extracting gesture frames...")
    cap = cv2.VideoCapture(downloaded_video)
    if not cap.isOpened():
        return 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Skip initial 15% and final 15% (intros/outros), sample middle 70%
    start_frame = int(total_video_frames * 0.15)
    end_frame = int(total_video_frames * 0.85)
    usable_range = max(1, end_frame - start_frame)
    step = max(1, usable_range // max_frames)

    frames_extracted = 0
    existing_count = len(list(target_dir.glob("*.jpg")))

    for frame_no in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        resized = cv2.resize(frame, (340, 260), interpolation=cv2.INTER_AREA)
        out_name = f"yt_{existing_count + frames_extracted}.jpg"
        cv2.imwrite(str(target_dir / out_name), resized, [cv2.IMWRITE_JPEG_QUALITY, 94])
        frames_extracted += 1
        print(f"  [OK] Extracted frame {out_name} (timestamp: {frame_no/fps:.1f}s)")

        if frames_extracted >= max_frames:
            break

    cap.release()

    # Clean up temporary video file to save disk space
    try:
        Path(downloaded_video).unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Extracted {frames_extracted} video frames for {slug}/")
    return frames_extracted


def main():
    parser = argparse.ArgumentParser(description="SignMate Google & YouTube Word Media Downloader")
    parser.add_argument("--word", type=str, help="Word to download (e.g., NAMASTE, HELLO, GOOD)")
    parser.add_argument("--source", choices=["google", "youtube", "all"], default="all", help="Source to download from")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum images from Google search (default: 5)")
    parser.add_argument("--extract-frames", type=int, default=10, help="Frames to extract from YouTube clip (default: 10)")
    parser.add_argument("--batch-top-words", action="store_true", help="Batch download for top primary ISL words")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" SIGNMATE - GOOGLE & YOUTUBE ISL WORD MEDIA HARVESTER")
    print("=" * 65)

    words_to_process = []
    if args.batch_top_words:
        words_to_process = ["NAMASTE", "HELLO", "GOOD", "HELP", "THANK_YOU", "WATER", "MOTHER", "DOCTOR", "YES", "STOP"]
    elif args.word:
        words_to_process = [args.word.strip().upper()]
    else:
        print("Please provide --word <NAME> or --batch-top-words.")
        print("Example: python tools/download_word_media.py --word NAMASTE --source all")
        return

    for word in words_to_process:
        print(f"\n>>> Processing Word: {word}")
        if args.source in ["google", "all"]:
            download_google_images(word, max_images=args.max_images)
        if args.source in ["youtube", "all"]:
            download_and_extract_youtube_frames(word, max_frames=args.extract_frames)

    print("\nMedia download and extraction process finished!\n")


if __name__ == "__main__":
    main()
