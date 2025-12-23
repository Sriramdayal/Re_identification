import cv2

def annotate(frame, boxes, ids):
    for (x1, y1, x2, y2), pid in zip(boxes.astype(int), ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame, pid, (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
        )
    return frame
