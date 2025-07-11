import pytesseract
import os
from PIL import Image

# Set path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Set TESSDATA_PREFIX so it knows where to find tessdata
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR"

# Check version
print(pytesseract.get_tesseract_version())

# Run test
img = Image.open("paper1.jpg")
print(pytesseract.image_to_string(img, lang="guj"))
