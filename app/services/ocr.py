# app/services/ocr.py
from pathlib import Path

import cv2
import pytesseract
from pdf2image import convert_from_path


def _ocr_image(image_path: str) -> str:
    """
    Run basic OpenCV preprocessing + Tesseract OCR on an image.
    """
    img = cv2.imread(image_path)

    if img is None:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise slightly
    gray = cv2.medianBlur(gray, 3)

    # You can experiment with thresholding to improve OCR for some docs:
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # text = pytesseract.image_to_string(thresh)

    text = pytesseract.image_to_string(gray)
    return text


def extract_text_from_file(path: str) -> str:
    """
    Handle PDF vs image. For PDFs, convert each page to an image, OCR each.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        pages = convert_from_path(path)
        all_text = []
        for i, page in enumerate(pages):
            temp_image = p.parent / f"{p.stem}_page_{i}.png"
            page.save(temp_image, "PNG")
            page_text = _ocr_image(str(temp_image))
            all_text.append(page_text)
            # Clean up temp image
            temp_image.unlink(missing_ok=True)
        return "\n".join(all_text)

    # Assume it's an image type otherwise
    return _ocr_image(path)
