
# 🎯 Player Re-Identification using YOLO + TorchReID

This project performs **person detection, jersey number OCR, and appearance-based re-identification** on football match videos using deep learning and computer vision.

---

## ✅ Features

- 🧠 Player detection using **YOLOv8**
- 🔢 OCR for jersey number extraction using `pytesseract`
- 🔁 Appearance feature extraction using **OSNet (TorchReID)**
- 🤝 Re-identification by hybrid matching: **OCR + Cosine Similarity**
- 📼 Video annotation with player bounding boxes and predicted IDs

---

## 🔧 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sriramdayal/Re_identification.git
    cd Re_identification
    ```

2.  **Set up the environment:**
    This project uses `uv` for dependency management, but standard pip works too.
    ```bash
    # Create and activate virtual environment
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    
    # Install the project in editable mode
    pip install -e .
    ```

    *Note: The `deep-person-reid` library is vendored in this repository to ensure compatibility and ease of setup.*

---

## � Usage

1.  **Prepare Input Video:**
    Place your input video in `data/input.mp4` (or update `configs/default.yaml`).

2.  **Run the Pipeline:**
    ```bash
    python3 scripts/run_pipeline.py
    ```

3.  **View Output:**
    The annotated video will be saved to `outputs/reid_output.mp4`.

---

## 📂 Project Structure

```
├── configs/               # Configuration files (YAML)
├── data/                  # Input data (videos)
├── deep-person-reid/      # Vendored TorchReID library
├── outputs/               # Generated results
├── scripts/               # Entry point scripts
│   └── run_pipeline.py    # Main pipeline script
├── src/
│   └── player_reid/       # Core package code
│       ├── detectors/     # YOLO detector
│       ├── embeddings/    # Appearance embeddings
│       ├── ocr/           # Jersey number OCR
│       ├── reid/          # Matching logic
│       └── video/         # Video processing utils
├── requirements.txt       # Python dependencies
└── setup.py               # Package installation
```

---

## 🛠️ Technologies Used
- YOLOv8 (Ultralytics)
- TorchReID (OSNet)
- PyTorch
- OpenCV
- Tesseract OCR
- Scikit-learn

---

## 👨‍💻 Author

Project by [Sriramdayal]
Adapted from TorchReID + Ultralytics YOLO

