from __future__ import annotations

import argparse
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_videos(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def safe_name(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text


@dataclass(frozen=True)
class ClassSpec:
    name: str
    path: Path


def extract_frames(
    video_path: Path,
    out_dir: Path,
    frames_per_video: int,
    target_size: tuple[int, int],
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        # Fallback: read sequentially until exhausted.
        total_frames = 0

    positions: list[int] = []
    if total_frames > 0:
        # Evenly spaced positions, avoiding very beginning/end.
        for i in range(frames_per_video):
            frac = (i + 1) / (frames_per_video + 1)
            pos = int(frac * total_frames)
            positions.append(max(0, min(total_frames - 1, pos)))
    else:
        positions = list(range(frames_per_video))

    saved = 0
    stem = safe_name(video_path.stem)
    for idx, pos in enumerate(positions):
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # BGR -> RGB not needed for saving jpg, but keep consistent.
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"{stem}_f{idx:02d}.jpg"
        ok = cv2.imwrite(str(out_path), frame_resized)
        if ok:
            saved += 1

    cap.release()
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare the DFD Kaggle dataset (video files) into an image-folder dataset: "
            "train/<class>/..., test/<class>/... by extracting frames." 
        )
    )
    p.add_argument("--input-root", default="data/kaggle_raw", help="Root where Kaggle extracted the dataset")
    p.add_argument("--real-dir", default="DFD_original sequences", help="Folder name for original videos")
    p.add_argument("--fake-dir", default="DFD_manipulated_sequences", help="Folder name for manipulated videos")
    p.add_argument(
        "--fake-nested",
        default="DFD_manipulated_sequences",
        help="If manipulated videos are nested in a same-named subfolder, set it here; empty to disable.",
    )
    p.add_argument("--output", default=".", help="Repo root output (default: current directory)")
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-videos-per-class", type=int, default=50)
    p.add_argument("--frames-per-video", type=int, default=2)
    p.add_argument("--size", type=int, default=224, help="Square image size to save")
    p.add_argument("--clean", action="store_true", help="Delete existing train/ and test/ folders before writing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output).resolve()

    real_root = input_root / args.real_dir
    fake_root = input_root / args.fake_dir
    if args.fake_nested:
        nested = fake_root / args.fake_nested
        if nested.exists():
            fake_root = nested

    if not real_root.exists():
        raise SystemExit(f"Real dir not found: {real_root}")
    if not fake_root.exists():
        raise SystemExit(f"Fake dir not found: {fake_root}")

    train_root = out_root / "train"
    test_root = out_root / "test"

    if args.clean:
        import shutil

        shutil.rmtree(train_root, ignore_errors=True)
        shutil.rmtree(test_root, ignore_errors=True)

    rng = random.Random(args.seed)

    specs = [
        ClassSpec("real", real_root),
        ClassSpec("fake", fake_root),
    ]

    target_size = (args.size, args.size)

    for spec in specs:
        videos = list(iter_videos(spec.path))
        if not videos:
            raise SystemExit(f"No videos found under {spec.path}")

        rng.shuffle(videos)
        if args.max_videos_per_class and args.max_videos_per_class > 0:
            videos = videos[: args.max_videos_per_class]

        n_test = max(1, int(round(len(videos) * args.test_split)))
        test_videos = videos[:n_test]
        train_videos = videos[n_test:]

        saved_train = 0
        saved_test = 0

        out_train = train_root / spec.name
        out_test = test_root / spec.name
        out_train.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)

        for v in train_videos:
            saved_train += extract_frames(v, out_train, args.frames_per_video, target_size)

        for v in test_videos:
            saved_test += extract_frames(v, out_test, args.frames_per_video, target_size)

        print(f"{spec.name}: videos train={len(train_videos)} test={len(test_videos)} | frames train={saved_train} test={saved_test}")

    print("\nPrepared frame dataset:")
    print(f"  {train_root}")
    print(f"  {test_root}")


if __name__ == "__main__":
    main()
