
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
!pip install -r requirements.txt
!python setup.py install
%cd /content
```

---

## 🚀 Setup & Execution (Colab)

1. **Upload your video to `/content/`**  
   Example: `your_input_video.mp4`

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


##  Project Objective
To detect and re-identify football players from match footage using object detection (YOLOv8), appearance-based embeddings (TorchReID), and jersey number OCR. The pipeline enables consistent tracking and annotation of players across frames, even with occlusions or partial visibility.

---

##  Approach: Step-by-Step

### 1.  Data Preparation
- Import video footage from Kaggle or Google Drive.
- Extract frames and crop individual player detections.

### 2.  Player Detection using YOLOv8
- Load YOLOv8 model (`yolov8n.pt`) using Ultralytics.
- Detect player bounding boxes in each frame (class `person`).
- Save each detected region as a cropped image.

### 3.  Jersey Number Recognition (OCR)
- Apply Tesseract OCR on each crop.
- Extract jersey numbers as rough player IDs.
- Store mapping of crop path → OCR text.

### 4. Appearance Embedding using TorchReID
- Load pretrained model `osnet_x1_0` from TorchReID.
- Extract 2048-dimensional feature vectors for each player crop.
- Store crop path → appearance embedding.

### 5.  Hybrid Matching Logic
- If a valid jersey number is detected (e.g., 7, 10), assign player ID as `jersey_7`, etc.
- If OCR fails or is uncertain:
  - Use cosine similarity with known embeddings.
  - Assign the closest match (if above threshold), else assign a temporary ID (`temp_0`, `temp_1`, ...).

### 6.  Annotated Video Generation
- Overlay bounding boxes and player IDs on original frames.
- Output a new video with consistent player annotations using OpenCV.

### 7. Optional: Embedding Classification
- Train a lightweight classifier (e.g., SVM or MLP) on embeddings to learn player identities for unseen frames.

---

## 🛠️ Technologies Used
- YOLOv8 (Ultralytics)
- TorchReID
- PyTorch
- OpenCV
- Tesseract OCR
- Scikit-learn (cosine similarity)

---

## ✅ Output
- Annotated MP4 video with real-time player labels.
- JSON or CSV results mapping crops to player IDs.

---



## 📽️ Sample Output

[https://drive.google.com/file/d/1tjCwz1O76C6Qjh_pFAePdyQm2dR5gtvV/view?usp=sharing]
colab notebook code reference:[https://colab.research.google.com/drive/1iZdCQRScsk52uUKledcjIfLdthugoxqP?usp=sharing]

---

## 👨‍💻 Author

Project by [Sriramdayal]
Adapted from TorchReID + Ultralytics YOLO

