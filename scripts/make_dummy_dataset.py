from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a tiny dummy dataset with train/test/prediction folders."
    )
    p.add_argument("--out", default=".", help="Repo root (default: current directory)")
    p.add_argument("--train", type=int, default=32, help="Images per class in train")
    p.add_argument("--test", type=int, default=16, help="Images per class in test")
    p.add_argument("--size", type=int, default=224, help="Image size (square)")
    return p.parse_args()


def save_noise_image(path: Path, size: int, tint: tuple[int, int, int]) -> None:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    arr = (0.75 * arr + 0.25 * np.array(tint, dtype=np.uint8)).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    args = parse_args()
    root = Path(args.out).resolve()

    # Class names can be anything; keep them intuitive.
    classes = {
        "real": (40, 200, 40),
        "fake": (200, 40, 40),
    }

    for split, count in [("train", args.train), ("test", args.test)]:
        for cls, tint in classes.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                save_noise_image(d / f"{cls}_{i:04d}.jpg", args.size, tint)

    (root / "prediction").mkdir(parents=True, exist_ok=True)
    save_noise_image(root / "prediction" / "sample_fake.jpg", args.size, classes["fake"])
    save_noise_image(root / "prediction" / "sample_real.jpg", args.size, classes["real"])

    print("Created dummy dataset at:", root)
    print("train/ test/ prediction/ are ready")


if __name__ == "__main__":
    main()
