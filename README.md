
# Sign Language Interpreter

A real-time sign language interpreter built with **OpenCV**, **MediaPipe**, and **scikit-learn**. It captures hand landmarks from a webcam, trains a classifier to recognize different signs, and then predicts signs live from your camera feed.

## How It Works

The project runs in three stages:

1. **Collect data** — capture webcam images for each sign/class you want to recognize.
2. **Build the dataset** — use MediaPipe to extract 21 hand landmarks (x, y) per image and save them as feature vectors.
3. **Train & run** — train a Random Forest classifier on the extracted landmarks, then use it to predict signs live from your webcam.

## Project Structure

```
Sign_language_interpreter/
├── collect_imgs.py           # Step 1: capture webcam images per class
├── create_dataset.py         # Step 2: extract hand landmarks, save data.pickle
├── train_classifier.py       # Step 3: train Random Forest, save model.p
├── inference_classifier.py   # Step 4: real-time webcam prediction
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Dishaagarwal-2006/Sign_language_interpreter.git
cd Sign_language_interpreter
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
```
Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in order:

**Step 1 — Collect images for each sign**
```bash
python collect_imgs.py
```
Opens your webcam. For each class, press `Q` when ready, then it captures a set number of frames automatically. Images are saved to `./data/<class_number>/`.

**Step 2 — Build the dataset**
```bash
python create_dataset.py
```
Runs MediaPipe hand detection on every collected image, extracts normalized landmark coordinates, and saves them to `data.pickle`.

**Step 3 — Train the classifier**
```bash
python train_classifier.py
```
Trains a `RandomForestClassifier` on the extracted landmarks, prints the test accuracy, and saves the trained model to `model.p`.

**Step 4 — Run real-time inference**
```bash
python inference_classifier.py
```
Opens your webcam, detects your hand, and shows the predicted sign live on screen. Press `Q` to quit.

⚠️ Before running this step, update the `labels_dict` in `inference_classifier.py` to map each class number to the actual sign it represents, e.g.:
```python
labels_dict = {0: 'Hello', 1: 'Thanks', 2: 'Yes'}
```

## Requirements

- Python 3.9–3.12
- A working webcam
- See `requirements.txt` for exact package versions

## Notes

- `data/` (raw images), `venv/`, and pickled files (`*.pickle`, `*.p`) are excluded from version control via `.gitignore` — you'll need to regenerate them locally by running the pipeline above.
- Accuracy shown during training reflects performance on a small, single-session dataset and may not generalize to different lighting, backgrounds, or hands. For better real-world performance, collect data across varied conditions.

## Tech Stack

- [OpenCV](https://opencv.org/) — webcam capture & image processing
- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [scikit-learn](https://scikit-learn.org/) — Random Forest classifier
