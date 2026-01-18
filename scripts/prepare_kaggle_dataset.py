from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a Kaggle image dataset into the folder structure expected by train.py: "
            "train/<class>/..., test/<class>/..."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help=(
            "Path to extracted Kaggle dataset. Expected layout: input/<class_name>/**.<img>. "
            "(If it contains train/ and test/ already, pass those explicitly with --input-train/--input-test.)"
        ),
    )
    p.add_argument("--output", default=".", help="Repo root output (default: current dir)")
    p.add_argument("--test-split", type=float, default=0.2, help="Fraction per class to put into test")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="If >0, limit total images per class before splitting (useful for quick smoke runs).",
    )
    p.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to place files into train/test. hardlink saves disk but may not work across drives.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing train/ and test/ folders in output before writing.",
    )
    p.add_argument(
        "--include-classes",
        default="",
        help=(
            "Comma-separated list of exactly 2 class folder names to include (binary classification). "
            "If omitted, the input must contain exactly 2 class folders."
        ),
    )
    return p.parse_args()


def hardlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            dst.hardlink_to(src)
            return
        except OSError:
            # Fall back to copy if hardlink fails.
            pass
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()

    if not input_root.exists():
        raise SystemExit(f"Input does not exist: {input_root}")

    train_root = out_root / "train"
    test_root = out_root / "test"

    if args.clean:
        shutil.rmtree(train_root, ignore_errors=True)
        shutil.rmtree(test_root, ignore_errors=True)

    # Class folders are immediate children of input_root.
    class_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise SystemExit(
            "Expected input to contain class subfolders, e.g. input/real and input/fake. "
            f"Found: {[p.name for p in class_dirs]}"
        )

    include = [c.strip() for c in args.include_classes.split(",") if c.strip()]
    if include:
        if len(include) != 2:
            raise SystemExit("--include-classes must list exactly 2 classes for binary classification")
        class_dirs = [p for p in class_dirs if p.name in set(include)]
        if len(class_dirs) != 2:
            raise SystemExit(
                f"Could not find both requested classes under input. Requested={include}, Found={[p.name for p in class_dirs]}"
            )
    else:
        if len(class_dirs) != 2:
            raise SystemExit(
                "This project is set up for binary classification (2 classes). "
                f"Found {len(class_dirs)} class folders: {[p.name for p in class_dirs]}. "
                "Re-run with --include-classes classA,classB."
            )

    rng = random.Random(args.seed)

    for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
        cls = class_dir.name
        images = list(iter_images(class_dir))
        if not images:
            print(f"Skipping empty class folder: {class_dir}")
            continue

        rng.shuffle(images)
        if args.max_per_class and args.max_per_class > 0:
            images = images[: args.max_per_class]
        n_test = max(1, int(round(len(images) * args.test_split)))
        test_images = images[:n_test]
        train_images = images[n_test:]

        for src in train_images:
            rel_name = f"{src.parent.name}_{src.name}"
            dst = train_root / cls / rel_name
            hardlink_or_copy(src, dst, args.mode)

        for src in test_images:
            rel_name = f"{src.parent.name}_{src.name}"
            dst = test_root / cls / rel_name
            hardlink_or_copy(src, dst, args.mode)

        print(f"{cls}: train={len(train_images)} test={len(test_images)}")

    print("\nPrepared dataset:")
    print(f"  train/: {train_root}")
    print(f"  test/ : {test_root}")
    print("\nNext: python .\\scripts\\train.py --train-dir train --test-dir test --epochs 50")


if __name__ == "__main__":
    main()
