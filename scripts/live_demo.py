from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live deepfake detection demo (webcam or video file).")
    p.add_argument("--model", default="models/cnn_deepfake.keras", help="Path to .keras model")
    p.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0) or a path to a video file",
    )
    p.add_argument("--size", type=int, default=224, help="Input size (square), default 224")
    p.add_argument("--threshold", type=float, default=0.5, help="Deepfake threshold")
    p.add_argument(
        "--every",
        type=int,
        default=3,
        help="Run model every N frames (higher = faster, less frequent predictions)",
    )
    p.add_argument(
        "--face",
        action="store_true",
        help="Try to detect a face and classify the face crop instead of full frame",
    )
    return p.parse_args()


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")
    return detector


def preprocess_bgr(img_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def pick_roi(frame_bgr: np.ndarray, face_detector: cv2.CascadeClassifier | None) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if face_detector is None:
        return frame_bgr, None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame_bgr, None

    # Pick largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + w + pad)
    y1 = min(frame_bgr.shape[0], y + h + pad)
    crop = frame_bgr[y0:y1, x0:x1]
    return crop, (x0, y0, x1 - x0, y1 - y0)


def draw_hud(frame_bgr: np.ndarray, text: str, prob: float | None, roi: tuple[int, int, int, int] | None) -> None:
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.rectangle(frame_bgr, (10, 10), (10 + 420, 10 + 70), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if prob is not None:
        cv2.putText(
            frame_bgr,
            f"prob_deepfake={prob:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    src: int | str
    src = int(args.source) if args.source.isdigit() else args.source

    model = tf.keras.models.load_model(str(model_path))
    face_detector = load_face_detector() if args.face else None

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    last_prob: float | None = None
    frame_idx = 0

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1

        roi_crop, roi = pick_roi(frame, face_detector)

        if frame_idx % max(1, args.every) == 0:
            x = preprocess_bgr(roi_crop, args.size)
            prob = float(model.predict(x, verbose=0)[0][0])
            last_prob = prob

        prob = last_prob
        if prob is None:
            label = "warming up..."
        else:
            label = "deepfake" if prob >= args.threshold else "real"

        draw_hud(frame, f"{label}  (thr={args.threshold:.2f})", prob, roi)
        cv2.imshow("Deepfake Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
