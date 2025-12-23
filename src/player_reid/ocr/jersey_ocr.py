import pytesseract
from PIL import Image

def read_jersey(image):
    text = pytesseract.image_to_string(
        Image.fromarray(image),
        config="--psm 6 digits"
    )
    text = text.strip()
    if text.isdigit():
        num = int(text)
        if 0 < num < 100:
            return num
    return None
