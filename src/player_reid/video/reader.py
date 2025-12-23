import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield fid, frame
        fid += 1
    cap.release()
