import cv2

def annotate(frame, boxes, ids):
    for (x1, y1, x2, y2), pid in zip(boxes.astype(int), ids):
        color = (0,255,0) if pid.startswith("jersey") else (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            frame, pid,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )
    return frame
