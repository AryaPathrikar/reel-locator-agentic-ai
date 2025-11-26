import os
import cv2
from typing import List


def extract_key_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 12,
) -> List[str]:
    """
    Robust frame extraction for real-world reels.

    - Works with variable frame rate
    - Uses evenly spaced sampling instead of frame_interval
    - Validates frame reads and writes
    - Tested on the user's reel.mp4
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)  # type: ignore
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    # Compute evenly spaced frame indexes
    step = max(1, total_frames // max_frames)
    frame_indexes = list(range(0, total_frames, step))[:max_frames]

    saved_paths: List[str] = []
    saved = 0

    for idx in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # type: ignore
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
        ok = cv2.imwrite(frame_path, frame)  # type: ignore
        if not ok:
            continue

        saved_paths.append(os.path.abspath(frame_path))
        saved += 1

    cap.release()

    if not saved_paths:
        raise RuntimeError(
            "Extraction failed. Video exists but no frames were readable. "
            "Try re-encoding the video or checking codec support."
        )

    return saved_paths
