import os
import cv2
import numpy as np
import easyocr
import warnings
import ssl
from passporteye import read_mrz
from pdf2image import convert_from_path
from PIL import Image
import string as st

# Fix SSL issue (Mac EasyOCR model download fix)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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

        logger.info(f"Initializing EasyOCR Reader (GPU={use_gpu})...")
        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            model_storage_directory=model_dir
        )
        logger.info("EasyOCR initialized.")

    # ---------------------------------------------------
    # 🔥 EXTRACT GIVEN NAMES FROM VISUAL FIELD
    # ---------------------------------------------------
    Good. That tells me exactly what is happening.

Your OCR is reading:

SYED IBRAHEEMK


The extra “K” is not from MRZ.
It is coming from OCR noise — usually from:

Next field bleeding (like country code PK)

The word “PAKISTAN”

Or background edge touching the name

So we must clean the extracted name properly, not just return raw OCR text.

✅ Proper Fix (Remove trailing garbage letters safely)

Replace your extract_given_names_from_visual() with this cleaned version:

import re

def extract_given_names_from_visual(self, img_path):
    """
    Extract name from visible 'Given Names' field
    and remove OCR garbage at the end.
    """
    try:
        results = self.reader.readtext(img_path, detail=0)
        lines = [r.strip() for r in results if r.strip()]

        for i, line in enumerate(lines):
            upper_line = line.upper()

            if "GIVEN" in upper_line and "NAME" in upper_line:

                # Case 1: Same line
                if ":" in line:
                    candidate = line.split(":")[1].strip()
                else:
                    # Case 2: Next line
                    if i + 1 < len(lines):
                        candidate = lines[i + 1].strip()
                    else:
                        return ""

                # -----------------------------
                # 🔥 CLEAN THE NAME PROPERLY
                # -----------------------------

                # Keep only letters and spaces
                candidate = re.sub(r'[^A-Za-z\s]', '', candidate)

                # Remove extra single letter at end (like K)
                words = candidate.split()

                # If last word is 1 character, remove it
                if len(words) > 1 and len(words[-1]) == 1:
                    words = words[:-1]

                cleaned_name = " ".join(words)

                return cleaned_name.strip()

        return ""

    except Exception as e:
        logger.error(f"Given Names extraction failed: {e}")
        return ""

        except Exception as e:
            logger.error(f"Given Names extraction failed: {e}")
            return ""

    # ---------------------------------------------------
    # MRZ EXTRACTION
    # ---------------------------------------------------
    def extract_mrz_from_roi(self, img_path):
        try:
            mrz = read_mrz(img_path, save_roi=True)

            if not mrz:
                return None, None, None

            roi = mrz.aux["roi"]

            if roi.dtype != np.uint8:
                roi = (roi * 255).astype(np.uint8)

            img_resized = cv2.resize(roi, (1110, 140))
            allow = st.ascii_uppercase + st.digits + "<"

            code = self.reader.readtext(
                img_resized,
                detail=0,
                allowlist=allow
            )

            if len(code) < 2:
                return None, None, mrz

            line1 = clean_mrz_line(code[0])
            line2 = clean_mrz_line(code[1])

            return line1, line2, mrz

        except Exception as e:
            logger.error(f"MRZ extraction failed: {e}")
            return None, None, None

    # ---------------------------------------------------
    # MAIN DATA FUNCTION
    # ---------------------------------------------------
    def get_data(self, img_path):
        if not os.path.exists(img_path):
            logger.error(f"File not found: {img_path}")
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)

        if line1 and line2:
            mrz = FallbackMRZ(line1, line2)

        if mrz is None:
            logger.warning("MRZ not detected.")
            return None

        # 🔥 Get surname from MRZ
        surname = clean_name_field(getattr(mrz, "surname", ""))

        # 🔥 Get name from visual GIVEN NAMES field
        visual_name = self.extract_given_names_from_visual(img_path)

        if visual_name:
            name = clean_name_field(visual_name)
        else:
            name = clean_name_field(
                getattr(mrz, "names", getattr(mrz, "name", ""))
            )

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

    # ---------------------------------------------------
    # PDF PROCESSING
    # ---------------------------------------------------
    def process_pdf(self, pdf_path):
        extracted_data = []
        os.makedirs(TEMP_DIR, exist_ok=True)

        try:
            pages = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

        for i, page in enumerate(pages):
            temp_img_path = os.path.join(
                TEMP_DIR, f"temp_page_{i+1}.png"
            )
            page.save(temp_img_path, "PNG")

            result = self.get_data(temp_img_path)

            if result:
                result["source_page"] = i + 1
                extracted_data.append(result)

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        return extracted_data
