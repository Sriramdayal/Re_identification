import yaml
import cv2

from player_reid.detectors.yolo_detector import PlayerDetector
from player_reid.embeddings.appearance import AppearanceEmbedder
from player_reid.ocr.jersey_ocr import read_jersey
from player_reid.reid.matcher import ReIDMatcher
from player_reid.video.reader import read_video
from player_reid.video.cropper import crop
from player_reid.video.annotator import annotate

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

detector = PlayerDetector(**cfg["detector"])
embedder = AppearanceEmbedder()
matcher = ReIDMatcher(cfg["reid"]["appearance_threshold"])

cap = cv2.VideoCapture(cfg["video"]["input"])
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

import os
os.makedirs(os.path.dirname(cfg["video"]["output"]), exist_ok=True)

out = cv2.VideoWriter(
    cfg["video"]["output"],
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (w, h)
)

for fid, frame in read_video(cfg["video"]["input"]):
    boxes = detector.detect(frame)
    crops = crop(frame, boxes)
    embeddings = embedder.embed(crops)

    ids = []
    for img, emb in zip(crops, embeddings):
        jersey = read_jersey(img)
        pid = matcher.assign(emb, jersey)
        ids.append(pid)

    annotated = annotate(frame, boxes, ids)
    out.write(annotated)

out.release()
print("✅ ReID complete")
