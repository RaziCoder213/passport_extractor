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

# Fix SSL issues (Mac + EasyOCR)
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
    clean_name_field,
    correct_mrz_common_errors,
    strict_mrz_filter
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
        logger.info("EasyOCR Ready.")

    # --------------------------------------------------
    # ROTATION FALLBACK
    # --------------------------------------------------
    def _retry_with_rotation(self, img_path):
        original = Image.open(img_path)

        for angle in [90, 180, 270]:
            rotated = original.rotate(angle, expand=True)
            temp_path = img_path + f"_rot_{angle}.png"
            rotated.save(temp_path)

            mrz = read_mrz(temp_path)

            os.remove(temp_path)

            if mrz:
                logger.info(f"MRZ found after rotation {angle}")
                return mrz

        return None

    # --------------------------------------------------
    # DIRECT EASY OCR FALLBACK
    # --------------------------------------------------
    def _fallback_direct_easyocr(self, img_path):
        result = self.reader.readtext(img_path, detail=0)

        potential = []
        for line in result:
            clean = clean_mrz_line(line)
            if len(clean) > 30 and ("<<" in clean or clean.startswith(("P<", "I<"))):
                potential.append(clean)

        if len(potential) >= 2:
            line1 = potential[-2]
            line2 = potential[-1]
            return line1, line2, FallbackMRZ(line1, line2)

        return None, None, None

    # --------------------------------------------------
    # MAIN MRZ EXTRACTION
    # --------------------------------------------------
    def extract_mrz(self, img_path):
        mrz = read_mrz(img_path)

        if not mrz:
            mrz = self._retry_with_rotation(img_path)

        if not mrz:
            return self._fallback_direct_easyocr(img_path)

        roi = mrz.aux["roi"]
        h = roi.shape[0]
        roi = roi[int(h * 0.6):h, :]

        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        roi = cv2.bilateralFilter(roi, 9, 75, 75)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)

        _, thresh = cv2.threshold(
            blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        scale = 140 / thresh.shape[0]
        resized = cv2.resize(thresh, (int(thresh.shape[1] * scale), 140))

        allow = st.ascii_uppercase + st.digits + "<"

        text = self.reader.readtext(
            resized,
            detail=0,
            allowlist=allow
        )

        if len(text) < 2:
            return None, None, mrz

        line1 = strict_mrz_filter(
            correct_mrz_common_errors(clean_mrz_line(text[0]))
        )
        line2 = strict_mrz_filter(
            correct_mrz_common_errors(clean_mrz_line(text[1]))
        )

        return line1, line2, FallbackMRZ(line1, line2)

    # --------------------------------------------------
    # DATA EXTRACTION
    # --------------------------------------------------
    def get_data(self, img_path):
        if not os.path.exists(img_path):
            return None

        line1, line2, mrz = self.extract_mrz(img_path)

        if not mrz:
            return None

        surname = clean_name_field(getattr(mrz, "surname", ""))
        name = clean_name_field(
            getattr(mrz, "names", getattr(mrz, "name", ""))
        )

        return {
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

    # --------------------------------------------------
    # PDF PROCESSING
    # --------------------------------------------------
    def process_pdf(self, pdf_path):
        os.makedirs(TEMP_DIR, exist_ok=True)

        pages = convert_from_path(pdf_path, dpi=300)
        results = []

        for i, page in enumerate(pages):
            temp_path = os.path.join(TEMP_DIR, f"page_{i}.png")
            page.save(temp_path, "PNG")

            data = self.get_data(temp_path)
            if data:
                data["source_page"] = i + 1
                results.append(data)

            os.remove(temp_path)

        return results
