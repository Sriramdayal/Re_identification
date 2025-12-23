import cv2
import numpy as np

class VisionAgent:
    @staticmethod
    def extract_crops(frame, boxes):
        """
        Extracts full-body and torso crops for each detection.
        Args:
            frame: Original video frame
            boxes: List of [x1, y1, x2, y2, conf]
        Returns:
            full_crops: List of full body images
            torso_crops: List of upper-body images (for OCR)
        """
        full_crops = []
        torso_crops = []
        
        h_img, w_img = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Clamp to frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            if x2 <= x1 or y2 <= y1:
                # Invalid box
                full_crops.append(None)
                torso_crops.append(None)
                continue

            # Full body crop
            crop = frame[y1:y2, x1:x2]
            full_crops.append(crop)

            # Torso crop (Upper 50%)
            # Ideally jerseys are in the upper half. 
            torso_h = int((y2 - y1) * 0.5)
            torso_crop = frame[y1 : y1+torso_h, x1:x2]
            torso_crops.append(torso_crop)

        return full_crops, torso_crops

    @staticmethod
    def check_quality(crop, min_size=(50, 100)):
        """
        Checks if a crop is suitable for Re-ID.
        """
        if crop is None:
            return False
        h, w = crop.shape[:2]
        if w < min_size[0] or h < min_size[1]:
            return False
        return True
