def crop(frame, boxes):
    crops = []
    for x1, y1, x2, y2 in boxes.astype(int):
        crops.append(frame[y1:y2, x1:x2])
    return crops
