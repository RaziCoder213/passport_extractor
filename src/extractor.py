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

# Fix SSL certificate issues (Mac + EasyOCR)
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

        logger.info(f"Initializing EasyOCR Reader (GPU={use_gpu})...")
        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            model_storage_directory=model_dir
        )
        logger.info("EasyOCR Reader initialized.")

    # --------------------------------------------------
    # ROTATION FALLBACK
    # --------------------------------------------------
    def _retry_with_rotation(self, img_path):
        try:
            original = Image.open(img_path)

            for angle in [90, 180, 270]:
                logger.info(f"Retrying with rotation: {angle}")
                rotated = original.rotate(angle, expand=True)

                temp_path = img_path + f"_rot_{angle}.png"
                rotated.save(temp_path)

                mrz = read_mrz(temp_path, save_roi=True)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if mrz:
                    logger.info(f"MRZ detected after rotation {angle}")
                    return mrz

            return None

        except Exception as e:
            logger.error(f"Rotation fallback failed: {e}")
            return None

    # --------------------------------------------------
    # DIRECT EASYOCR FALLBACK
    # --------------------------------------------------
    def _fallback_direct_easyocr(self, img_path):
        try:
            result = self.reader.readtext(img_path, detail=0)

            potential_lines = []
            for line in result:
                clean = clean_mrz_line(line)
                if len(clean) > 30 and (
                    "<<" in clean or clean.startswith(("P<", "I<", "A<", "V<"))
                ):
                    potential_lines.append(clean)

            if len(potential_lines) >= 2:
                line1 = potential_lines[-2]
                line2 = potential_lines[-1]

                if not line1.startswith(("P<", "I<", "A<", "V<")):
                    for i, l in enumerate(potential_lines):
                        if l.startswith(("P<", "I<", "A<", "V<")) and i + 1 < len(potential_lines):
                            line1 = l
                            line2 = potential_lines[i + 1]
                            break

                logger.info("Direct EasyOCR found potential MRZ")
                return line1, line2, FallbackMRZ(line1, line2)

            return None, None, None

        except Exception as e:
            logger.error(f"Direct EasyOCR fallback failed: {e}")
            return None, None, None

    # --------------------------------------------------
    # MRZ EXTRACTION
    # --------------------------------------------------
    def extract_mrz_from_roi(self, img_path):
        try:
            mrz = read_mrz(img_path, save_roi=True)

            if not mrz:
                logger.warning("PassportEye failed. Trying rotation...")
                mrz = self._retry_with_rotation(img_path)

            if not mrz:
                logger.warning("Rotation failed. Using full image OCR fallback...")
                return self._fallback_direct_easyocr(img_path)

            roi = mrz.aux["roi"]
            h, w = roi.shape[:2]

            # Crop bottom portion
            roi = roi[int(h * 0.6):h, :]

            if roi.dtype != np.uint8:
                if roi.max() <= 1.0:
                    roi = (roi * 255).astype(np.uint8)
                else:
                    roi = roi.astype(np.uint8)

            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            denoised = cv2.bilateralFilter(gray, 9, 75, 75)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
            blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

            norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

            _, thresh = cv2.threshold(
                norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            small_kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)

            scale = 140 / cleaned.shape[0]
            resized = cv2.resize(
                cleaned,
                (int(cleaned.shape[1] * scale), 140)
            )

            allow = st.ascii_uppercase + st.digits + "<"

            code = self.reader.readtext(
                resized,
                detail=0,
                allowlist=allow,
                paragraph=False,
                width_ths=0.5,
                height_ths=0.5
            )

            if len(code) < 2:
                logger.warning("EasyOCR found fewer than 2 lines.")
                return None, None, mrz

            line1 = strict_mrz_filter(
                correct_mrz_common_errors(clean_mrz_line(code[0]))
            )
            line2 = strict_mrz_filter(
                correct_mrz_common_errors(clean_mrz_line(code[1]))
            )

            return line1, line2, FallbackMRZ(line1, line2)

        except Exception as e:
            logger.error(f"MRZ extraction failed: {e}")
            return None, None, None

    # --------------------------------------------------
    # DATA EXTRACTION
    # --------------------------------------------------
    def get_data(self, img_path, airline=None):
        if not os.path.exists(img_path):
            logger.error(f"File not found: {img_path}")
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)

        if line1 and line2:
            mrz = FallbackMRZ(line1, line2)

        if not mrz:
            logger.warning("No valid MRZ extracted.")
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
    def process_pdf(self, pdf_path, airline=None):
        os.makedirs(TEMP_DIR, exist_ok=True)

        try:
            pages = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

        results = []

        for i, page in enumerate(pages):
            temp_path = os.path.join(TEMP_DIR, f"page_{i}.png")
            page.save(temp_path, "PNG")

            data = self.get_data(temp_path, airline=airline)

            if data:
                data["source_page"] = i + 1
                results.append(data)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        return results
