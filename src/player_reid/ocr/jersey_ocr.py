import pytesseract
from PIL import Image

def extract_jersey(crop):
    text = pytesseract.image_to_string(
        Image.fromarray(crop),
        config="--psm 6 digits"
    ).strip()

    if text.isdigit():
        num = int(text)
        if 0 < num < 100:
            return num
    return None
