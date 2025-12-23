from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model: str, conf: float):
        self.model = YOLO(model)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(
            frame, conf=self.conf, classes=[0], verbose=False
        )
        return results[0].boxes.xyxy.cpu().numpy()
