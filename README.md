
# 🎯 Player Re-Identification using YOLO + TorchReID

This project performs **person detection, jersey number OCR, and appearance-based re-identification** on football match videos using deep learning and computer vision.

---

## 📁 Project Structure

```
.
├── Re-id.ipynb              # Main Colab notebook
├── crops/                   # Extracted player crops per frame
├── annotated_crops/         # Crops annotated with predicted IDs
├── reid_output.mp4          # Final annotated output video
├── reid_results.json        # Player ID assignments (OCR + appearance)
├── data.pkl                 # Optional: pretrained model checkpoint or data embeddings
├── README.md                # This file
```

---

## ✅ Features

- 🧠 Player detection using **YOLOv8**
- 🔢 OCR for jersey number extraction using `pytesseract`
- 🔁 Appearance feature extraction using **OSNet (TorchReID)**
- 🤝 Re-identification by hybrid matching: **OCR + Cosine Similarity**
- 📼 Video annotation with player bounding boxes and predicted IDs

---

## 🔧 Requirements

Install the following:

```bash
pip install ultralytics
pip install torch torchvision torchaudio
pip install opencv-python
pip install scikit-learn
pip install pytesseract
pip install torchreid
```

⚠️ If using **Google Colab**, run this to install `torchreid`:

```python
!git clone https://github.com/KaiyangZhou/deep-person-reid.git
%cd deep-person-reid
!pip install -e .
```

---

## 🚀 Setup & Execution (Colab)

1. **Upload your video to `/content/`**  
   Example: `tacticam.mp4`

2. **Run the Notebook** step by step:
   - Extract player crops per frame using YOLO
   - Perform OCR on each crop
   - Extract embeddings using TorchReID
   - Match OCR and embeddings
   - Annotate and export the final video

---

## 📚 Modules & Libraries Imported

- `torch`, `torchvision` — model loading, tensor operations
- `ultralytics` — YOLOv8 detection
- `opencv-python (cv2)` — image & video processing
- `pytesseract` — OCR for jersey numbers
- `sklearn.metrics.pairwise` — cosine similarity
- `torchreid` — feature extractor for appearance embeddings
- `matplotlib`, `json`, `glob`, `os`, `shutil` — utilities

---

## 📈 Hybrid Re-Identification

Players are matched using:
1. **OCR match:** If the jersey number is clear
2. **Appearance match:** Cosine similarity of embeddings
3. **Fallback:** Temporary IDs assigned if no good match found

---

## 📝 Notes

- Works best on high-resolution videos with visible jersey numbers.
- OSNet model is pretrained on ImageNet and optimized for re-ID tasks.
- `.pkl` checkpoints can optionally store embeddings or models.

---

## 📽️ Sample Output

![Output](https://i.imgur.com/xBcdP0H.gif)

---

## 👨‍💻 Author

Project by [Sriramdayal]
Adapted from TorchReID + Ultralytics YOLO