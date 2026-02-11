import os
import cv2
import numpy as np
import easyocr
import warnings
import ssl
import re
from passporteye import read_mrz
from pdf2image import convert_from_path
from PIL import Image
import string as st

from src.utils import (
    clean_string,
    clean_mrz_line,
    parse_date,
    get_country_name,
    get_sex,
    setup_logger,
    clean_name_field
)

from src.fallback_mrz import FallbackMRZ
from config.settings import USE_GPU, OCR_LANGUAGES, TEMP_DIR

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)


class PassportExtractor:

    def __init__(self, use_gpu=USE_GPU, languages=None):
        self.languages = languages if languages else OCR_LANGUAGES

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "data", "models")
        os.makedirs(model_dir, exist_ok=True)

        logger.info(f"Initializing EasyOCR (GPU={use_gpu})...")
        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            model_storage_directory=model_dir
        )
        logger.info("EasyOCR initialized.")


    # ================= MRZ EXTRACTION ================= #

    def extract_mrz_from_roi(self, img_path):
        try:
            mrz = read_mrz(img_path, save_roi=True)

            if not mrz:
                logger.warning("MRZ not detected.")
                return None, None, None

            roi = mrz.aux["roi"]

            if roi.dtype != np.uint8:
                if roi.max() <= 1.0:
                    roi = (roi * 255).astype(np.uint8)
                else:
                    roi = roi.astype(np.uint8)

            roi = cv2.resize(roi, (1110, 140))

            allow = st.ascii_uppercase + st.digits + "<"

            lines = self.reader.readtext(
                roi,
                detail=0,
                allowlist=allow
            )

            if len(lines) < 2:
                return None, None, mrz

            line1 = clean_mrz_line(lines[0])
            line2 = clean_mrz_line(lines[1])

            return line1, line2, mrz

        except Exception as e:
            logger.error(f"MRZ extraction error: {e}")
            return None, None, None


    # ================= VISUAL GIVEN NAME EXTRACTION ================= #

    def extract_given_name_from_visual_zone(self, img_path):
        try:
            results = self.reader.readtext(img_path)

            for (bbox, text, conf) in results:
                if conf < 0.5:
                    continue

                text_upper = text.upper().strip()

                # Look for GIVEN NAME label
                if re.search(r"GIVEN\s+NAME", text_upper):

                    # Case 1: Same line
                    if ":" in text_upper:
                        name = text_upper.split(":", 1)[1].strip()
                        return clean_name_field(name)

                    # Case 2: Next OCR line
                    index = results.index((bbox, text, conf))
                    if index + 1 < len(results):
                        next_text = results[index + 1][1].strip()
                        return clean_name_field(next_text)

            return None

        except Exception as e:
            logger.error(f"Visual name extraction error: {e}")
            return None


    # ================= MAIN DATA FUNCTION ================= #

    def get_data(self, img_path):

        if not os.path.exists(img_path):
            logger.error("File not found.")
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)

        if line1 and line2:
            mrz = FallbackMRZ(line1, line2)

        if not mrz:
            logger.warning("No valid MRZ found.")
            return None

        # -------- NAME PRIORITY FIX -------- #
        visual_given_name = self.extract_given_name_from_visual_zone(img_path)

        if visual_given_name and len(visual_given_name) > 1:
            name = clean_name_field(visual_given_name)
            logger.info(f"Using GIVEN NAME from visual field: {name}")
        else:
            name = clean_name_field(
                getattr(mrz, "names", getattr(mrz, "name", ""))
            )
            logger.warning("Using MRZ name as fallback.")

        surname = clean_name_field(getattr(mrz, "surname", ""))

        data = {
            "surname": surname,
            "name": name,
            "country": get_country_name(getattr(mrz, "country", "")),
            "nationality": get_country_name(getattr(mrz, "nationality", "")),
            "passport_number": clean_string(getattr(mrz, "number", "")),
            "sex": get_sex(getattr(mrz, "sex", "")),
            "date_of_birth": parse_date(getattr(mrz, "date_of_birth", "")),
            "expiration_date": parse_date(getattr(mrz, "expiration_date", "")),
            "mrz_full_string": (line1 or "") + (line2 or ""),
            "valid_score": getattr(mrz, "valid_score", 0),
        }

        return data


    # ================= PDF PROCESSING ================= #

    def process_pdf(self, pdf_path):

        os.makedirs(TEMP_DIR, exist_ok=True)

        extracted_data = []

        try:
            pages = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

        for i, page in enumerate(pages):
            temp_img = os.path.join(TEMP_DIR, f"page_{i+1}.png")
            page.save(temp_img, "PNG")

            result = self.get_data(temp_img)

            if result:
                result["source_page"] = i + 1
                extracted_data.append(result)

            if os.path.exists(temp_img):
                os.remove(temp_img)

        return extracted_data
