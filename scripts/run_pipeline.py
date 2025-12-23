import yaml
import cv2
import os
import sys
import numpy as np

from player_reid.detectors.yolo_detector import PlayerDetector
from player_reid.vision.cropper import VisionAgent
from player_reid.tracking.byte_tracker import BytesTrackAgent
from player_reid.ocr.jersey_ocr import extract_jersey
from player_reid.ocr.manager import OCRManager
from player_reid.embeddings.appearance import AppearanceEmbedder
from player_reid.reid.manager import ReIDManager
from player_reid.video.reader import read_video
from player_reid.video.annotator import annotate

def main():
    # 1. Load Config
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    # 2. Initialize Agents
    detector = PlayerDetector(**cfg["detector"])
    tracker = BytesTrackAgent(frame_rate=30) # Default to 30, can be parsed from video
    embedder = AppearanceEmbedder()
    
    ocr_manager = OCRManager()
    reid_manager = ReIDManager(cfg["reid"]["appearance_threshold"])

    # 3. Setup Video
    cap = cv2.VideoCapture(cfg["video"]["input"])
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {cfg['video']['input']}")
        sys.exit(1)
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"✅ Input video found: {cfg['video']['input']} ({w}x{h} @ {fps:.2f} fps)")

    os.makedirs(os.path.dirname(cfg["video"]["output"]), exist_ok=True)
    out = cv2.VideoWriter(
        cfg["video"]["output"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    # 4. Pipeline Execution
    for fid, frame in read_video(cfg["video"]["input"]):
        # A. Vision Agent: Detect
        # Raw Detections: [x1, y1, x2, y2, conf]
        raw_dets = detector.detect(frame)

        # B. Tracking Agent: Assign Track IDs
        # Tracks: [x1, y1, x2, y2, track_id, conf, cls] (Format varies by tracker, assuming this)
        tracks = tracker.update(raw_dets, frame.shape[:2])
        
        # Parse tracks into Standard Format for downstream
        # We need boxes with track_ids for cropping
        # Format: list of [x1, y1, x2, y2, track_id]
        active_tracks = []
        for t in tracks:
            # Ultralytics ByteTrack output is often [x1, y1, x2, y2, id, conf, cls]
            # Verify shape/indices based on library version. 
            # Assuming standard output here.
            tlwh = t[:4]
            tid = int(t[4])
            active_tracks.append([*tlwh, tid])

        if len(active_tracks) == 0:
            out.write(frame)
            continue

        # C. Vision Agent: Crop (Full + Torso)
        # We only care about active tracks now
        track_boxes = [t[:4] for t in active_tracks]
        full_crops, torso_crops = VisionAgent.extract_crops(frame, track_boxes)
        
        # D. Feature Extraction
        # Filter valid crops
        valid_indices = [i for i, c in enumerate(full_crops) if VisionAgent.check_quality(c)]
        
        # Prepare batch for embedding
        batch_crops = [full_crops[i] for i in valid_indices]
        batch_embeddings = embedder.embed(batch_crops) if batch_crops else []

        final_ids = []
        
        # E. Logic Loop per Track
        # We iterate through ALL tracks, but only process those with valid crops/embeddings
        embed_idx = 0
        
        for i, track_data in enumerate(active_tracks):
            track_id = track_data[4]
            # Default display ID is the track ID until resolved
            display_id = f"ID_{track_id}"
            
            if i in valid_indices:
                # 1. Update Appearance
                emb = batch_embeddings[embed_idx]
                reid_manager.update_track_embedding(track_id, emb)
                embed_idx += 1
                
                # 2. Update OCR
                # Only run OCR if torso crop is good
                if VisionAgent.check_quality(torso_crops[i], min_size=(20, 20)):
                    jersey = extract_jersey(torso_crops[i])
                    ocr_manager.update(track_id, jersey)
                    
                # 3. Resolve Identity
                # Check for stable jersey
                stable_jersey = ocr_manager.get_stable_jersey(track_id)
                # Resolve final ID
                resolved_id = reid_manager.resolve_identity(track_id, stable_jersey)
                display_id = resolved_id
            
            else:
                # If quality was bad, just reuse last known ID for this track
                resolved_id = reid_manager.resolve_identity(track_id, None)
                display_id = resolved_id

            final_ids.append(display_id)

        # F. Annotation
        # Annotator expects [x1,y1,x2,y2] and [ids]
        boxes_only = np.array(track_boxes)
        annotated = annotate(frame, boxes_only, final_ids)
        out.write(annotated)

        if fid % 30 == 0:
            print(f"Processed frame {fid}")

    out.release()
    print("✅ Pipeline complete. Saved to:", cfg["video"]["output"])

if __name__ == "__main__":
    main()
