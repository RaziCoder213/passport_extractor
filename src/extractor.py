import os
import cv2
import numpy as np
import easyocr
import warnings
import ssl
import re
from passporteye import read_mrz
from pdf2image import convert_from_path
import string as st

from src.utils import (
    clean_string,
    clean_mrz_line,
    parse_date,
    format_date_flydubai,
    get_country_name,
    get_sex,
    setup_logger,
    clean_name_field
)

from src.fallback_mrz import FallbackMRZ
from config.settings import USE_GPU, OCR_LANGUAGES, TEMP_DIR

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)


# Fix SSL issue (Mac EasyOCR model download fix)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


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
    # VISUAL GIVEN NAME EXTRACTION (PRIMARY SOURCE)
    # ---------------------------------------------------
    def extract_given_names_from_visual(self, img_path):
        try:
            results = self.reader.readtext(img_path, detail=0)
            lines = [r.strip() for r in results if r.strip()]

            for i, line in enumerate(lines):
                upper_line = line.upper()

                if "GIVEN" in upper_line and "NAME" in upper_line:

                    if ":" in line:
                        candidate = line.split(":")[1].strip()
                    else:
                        if i + 1 < len(lines):
                            candidate = lines[i + 1].strip()
                        else:
                            return ""

                    # Keep only letters and spaces
                    candidate = re.sub(r'[^A-Za-z\s]', '', candidate)

                    # Remove trailing single letter (e.g. IBRAHEEMK → IBRAHEEM)
                    candidate = re.sub(r'([A-Z]{3,})[A-Z]$', r'\1', candidate)

                    return candidate.strip()

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

        surname = clean_name_field(getattr(mrz, "surname", ""))

        # 🔥 ALWAYS prefer visual name
        visual_name = self.extract_given_names_from_visual(img_path)

        if visual_name:
            name = visual_name
        else:
            name = clean_name_field(
                getattr(mrz, "names", getattr(mrz, "name", ""))
            )

            # Final defensive cleanup
            name = re.sub(r'([A-Z]{3,})[A-Z]$', r'\1', name)

        data = {
            "surname": surname,
            "name": name,
            "country": get_country_name(getattr(mrz, "country", "")),
            "nationality": get_country_name(getattr(mrz, "nationality", "")),
            "passport_number": clean_string(getattr(mrz, "number", "")),
            "sex": get_sex(getattr(mrz, "sex", "")),
            "date_of_birth": format_date_flydubai(getattr(mrz, "date_of_birth", "")),
            "expiration_date": format_date_flydubai(getattr(mrz, "expiration_date", "")),
            "mrz_full_string": (line1 or "") + (line2 or ""),
            "valid_score": getattr(mrz, "valid_score", 0),
        }

        return data
