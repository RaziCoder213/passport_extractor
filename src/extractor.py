# passport_extractor.py
import os
import cv2
import numpy as np
import easyocr
from passporteye import read_mrz
from pdf2image import convert_from_path
from PIL import Image
from utils import (
    setup_logger,
    clean_name_field,
    clean_mrz_line,
    parse_date,
    parse_mrz_names
)
import string as st

TEMP_DIR = "temp_passport_images"

logger = setup_logger(__name__)

class PassportExtractor:
    def __init__(self, use_gpu=False, languages=['en']):
        self.languages = languages
        self.reader = easyocr.Reader(self.languages, gpu=use_gpu)

    def extract_mrz(self, img_path):
        """Extract MRZ lines using PassportEye, fallback to EasyOCR"""
        mrz = read_mrz(img_path)
        if mrz:
            line1 = clean_mrz_line(mrz.lines[0])
            line2 = clean_mrz_line(mrz.lines[1])
            return line1, line2, mrz
        # Fallback with EasyOCR
        logger.info("PassportEye failed, using EasyOCR fallback...")
        allow_chars = st.ascii_uppercase + st.digits + "<"
        result = self.reader.readtext(img_path, detail=0, allowlist=allow_chars)
        potential_lines = [clean_mrz_line(r) for r in result if '<<' in r and len(r) >= 30]
        if len(potential_lines) >= 2:
            return potential_lines[-2], potential_lines[-1], None
        return None, None, None

    def get_passport_data(self, img_path):
        if not os.path.exists(img_path):
            logger.error(f"File not found: {img_path}")
            return None
        line1, line2, mrz_obj = self.extract_mrz(img_path)
        if not line1 or not line2:
            logger.warning(f"MRZ extraction failed for {img_path}")
            return None

        # Parse names from line1 only
        surname, given_names = parse_mrz_names(line1)
        surname = clean_name_field(surname)
        given_names = clean_name_field(given_names)

        # MRZ object data fallback if available
        passport_number = mrz_obj.number if mrz_obj else ""
        nationality = mrz_obj.nationality if mrz_obj else ""
        country = mrz_obj.country if mrz_obj else ""
        sex = mrz_obj.sex if mrz_obj else ""
        dob = parse_date(mrz_obj.date_of_birth) if mrz_obj else ""
        expiry = parse_date(mrz_obj.expiration_date) if mrz_obj else ""

        data = {
            "surname": surname,
            "given_names": given_names,
            "passport_number": passport_number,
            "nationality": nationality,
            "country": country,
            "sex": sex,
            "date_of_birth": dob,
            "expiration_date": expiry,
            "mrz_line1": line1,
            "mrz_line2": line2
        }
        return data

    def process_pdf(self, pdf_path):
        """Extract passport data from each page of a PDF"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return []

        os.makedirs(TEMP_DIR, exist_ok=True)
        try:
            pages = convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

        results = []
        for i, page in enumerate(pages):
            temp_img_path = os.path.join(TEMP_DIR, f"page_{i+1}.png")
            page.save(temp_img_path, "PNG")
            data = self.get_passport_data(temp_img_path)
            if data:
                results.append(data)
            os.remove(temp_img_path)
        return results

# Example usage
if __name__ == "__main__":
    extractor = PassportExtractor(use_gpu=False)
    image_path = "passport.jpg"  # Replace with your image
    data = extractor.get_passport_data(image_path)
    print(data)
