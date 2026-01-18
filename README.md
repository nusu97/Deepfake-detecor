# Deepfake-detecor

Implements the TechVidvan guide “DeepFake Detection using Convolutional Neural Networks”.

The tutorial assumes a folder-based dataset with three top-level directories:

- `train/` (training images)
- `test/` (validation/testing images)
- `prediction/` (some sample images to try prediction)

Each of `train/` and `test/` must contain **two class subfolders** (names can be anything), for example:

```
train/
	real/
	fake/
test/
	real/
	fake/
prediction/
	deepfake1.jpg
```

## Setup (Windows / PowerShell)

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download the tutorial dataset

The original TechVidvan dataset zip may be unavailable or blocked (403). A Kaggle dataset is usually more reliable.

### Kaggle dataset (recommended)

1) Install Kaggle CLI and set credentials

- `pip install kaggle`
- Put your Kaggle API token at either:
  - `%USERPROFILE%\.kaggle\kaggle.json` (standard Kaggle location)
  - `.kaggle/kaggle.json` (repo-local; this repo ignores `.kaggle/` so you don't commit secrets)
	(Download it from Kaggle: Account → API → “Create New API Token”)

2) Download a dataset (replace `<owner/dataset>` with the Kaggle slug)

```powershell
.\scripts\kaggle_download.ps1 -Dataset <owner/dataset>
```

Or run an end-to-end smoke test (download → prepare small subset → train 1 epoch → predict):

```powershell
.\scripts\run_kaggle_smoke.ps1 -Dataset sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset -MaxPerClass 200 -Epochs 1
```

Note: that specific dataset is **video-based** (it contains `.mp4` files). The smoke script automatically extracts a couple frames per video into `train/` and `test/` before training.

3) Prepare into `train/` + `test/`

Point the script to the extracted folder that contains class subfolders (e.g. `real/` and `fake/`):

```powershell
python .\scripts\prepare_kaggle_dataset.py --input .\data\kaggle_raw --output . --clean
```

If your dataset has more than 2 class folders, select exactly 2 for binary classification:

```powershell
python .\scripts\prepare_kaggle_dataset.py --input <folder> --include-classes real,fake --output . --clean
```

### Video datasets (DFD-style)

If the Kaggle dataset contains videos (e.g. the DFD dataset above), use frame extraction:

```powershell
python .\scripts\prepare_dfd_videos.py --input-root .\data\kaggle_raw --clean --max-videos-per-class 200 --frames-per-video 2
python .\scripts\train.py --train-dir train --test-dir test --epochs 50 --out models\cnn_deepfake.keras
```

If your Kaggle dataset extracts into a nested folder, pass that nested folder to `--input`.

### TechVidvan zip (optional)

If you do have the zip, place it at repo root as `deep-fake-detection.zip` and run:

```powershell
.\scripts\extract_dataset.ps1
```

## Train

```powershell
python .\scripts\train.py --train-dir train --test-dir test --epochs 50 --out models\cnn_deepfake.keras
```

### Real training (DFD Kaggle videos)

For the DFD Kaggle dataset (videos), run this to extract frames and train:

```powershell
.\scripts\run_kaggle_train.ps1 -MaxVideosPerClass 200 -FramesPerVideo 3 -Epochs 10
```

## Predict (single image)

```powershell
python .\scripts\predict.py --model models\cnn_deepfake.keras .\prediction\deepfake1.jpg
```

Output looks like:

`prediction=deepfake prob_deepfake=0.8123 file=...`

## Live demo (webcam / video)

Webcam (press `q` to quit):

```powershell
python .\scripts\live_demo.py --model .\models\cnn_deepfake.keras --source 0 --face
```

Video file:

```powershell
python .\scripts\live_demo.py --model .\models\cnn_deepfake.keras --source .\path\to\video.mp4 --face
```

Tips:

- Use `--every 5` to run the model every 5 frames (faster UI).
- Remove `--face` to classify the full frame (less accurate, but works if face detection fails).

## Notes

- Model architecture intentionally mirrors the tutorial (see `src/deepfake_detector/model.py`).
- If you want better accuracy, the next step is usually: more data, face-cropping, and a stronger backbone (e.g., EfficientNet). This repo sticks to the tutorial CNN on purpose.