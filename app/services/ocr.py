# app/services/ocr.py
from pathlib import Path

import cv2
import pytesseract
from pdf2image import convert_from_path

# Your Poppler path on Windows
POPPLER_PATH = r"C:\poppler-25.11.0\Library\bin"


def _ocr_image(image_path: str) -> str:
    """
    Run OpenCV preprocessing + Tesseract OCR on an image.
    """
    img = cv2.imread(image_path)

    if img is None:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Slight denoising
    gray = cv2.medianBlur(gray, 3)

    # Optional: adaptive thresholding to improve contrast
    # gray = cv2.adaptiveThreshold(
    #     gray, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     31, 2
    # )

    # Better Tesseract config
    config = "--oem 3 --psm 6"  # LSTM engine, assume block of text

    text = pytesseract.image_to_string(gray, config=config)
    return text


def extract_text_from_file(path: str) -> str:
    """
    Handle PDF vs image. For PDFs, convert each page to an image with higher DPI.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        # Use higher DPI for clearer text for OCR
        pages = convert_from_path(path, dpi=300, poppler_path=POPPLER_PATH)
        all_text = []
        for i, page in enumerate(pages):
            temp_image = p.parent / f"{p.stem}_page_{i}.png"
            page.save(temp_image, "PNG")
            page_text = _ocr_image(str(temp_image))
            all_text.append(page_text)
            temp_image.unlink(missing_ok=True)
        return "\n".join(all_text)

    # Assume image otherwise
    return _ocr_image(path)
