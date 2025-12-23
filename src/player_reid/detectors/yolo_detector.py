from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model="yolov8n.pt", conf=0.4):
        self.model = YOLO(model)
        self.conf = conf

    def detect(self, frame):
        """
        Detects players in a frame.
        Returns:
            detections (list): List of [x1, y1, x2, y2, conf]
        """
        results = self.model.predict(
            frame, conf=self.conf, classes=[0], verbose=False
        )
        
        # Return boxes and confidence: [x1, y1, x2, y2, conf]
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            return [ [*b, c] for b, c in zip(boxes, confs) ]
        
        return []
