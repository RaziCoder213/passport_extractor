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

# Fix SSL issues (Mac / Streamlit Cloud)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Import Utils
from src.utils import (
    clean_string,
    clean_mrz_line,
    parse_date,
    get_country_name,
    get_sex,
    setup_logger
)
# Import Validator
from src.validator import validate_passport_data
from src.fallback_mrz import FallbackMRZ
from config.settings import USE_GPU, OCR_LANGUAGES, TEMP_DIR

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

class PassportExtractor:
    """
    Passport MRZ extractor using PassportEye + EasyOCR
    Compatible with CLI and Streamlit
    """

    def __init__(self, use_gpu=USE_GPU, languages=None, airline=None):
        self.use_gpu = use_gpu
        self.languages = languages if languages else OCR_LANGUAGES
        self.airline = airline  # "iraq" or "flydubai"

        # Model storage directory (fixes Streamlit permissions)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "data", "models")
        os.makedirs(model_dir, exist_ok=True)

        logger.info(f"Initializing EasyOCR Reader (GPU={use_gpu}, airline={airline})")

        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            model_storage_directory=model_dir
        )

        logger.info("EasyOCR Reader initialized successfully")

    def _retry_with_rotation(self, img_path):
        try:
            original = Image.open(img_path)
            for angle in [90, 180, 270]:
                rotated = original.rotate(angle, expand=True)
                temp_path = f"{img_path}_rot_{angle}.png"
                rotated.save(temp_path)
                mrz = read_mrz(temp_path, save_roi=True)
                os.remove(temp_path)
                if mrz:
                    logger.info(f"MRZ found after rotation {angle}")
                    return mrz
            return None
        except Exception as e:
            logger.error(f"Rotation retry failed: {e}")
            return None

    def _fallback_direct_easyocr(self, img_path):
        try:
            result = self.reader.readtext(img_path, detail=0)
            candidates = []
            for line in result:
                clean = clean_mrz_line(line)
                if len(clean) > 30 and (clean.startswith(("P<", "I<", "A<", "V<")) or "<<" in clean):
                    candidates.append(clean)
            if len(candidates) >= 2:
                line1, line2 = candidates[-2], candidates[-1]
                logger.info("MRZ recovered via EasyOCR fallback")
                return line1, line2, FallbackMRZ(line1, line2)
            return None, None, None
        except Exception as e:
            logger.error(f"EasyOCR fallback failed: {e}")
            return None, None, None

    def extract_mrz_from_roi(self, img_path):
        try:
            mrz = read_mrz(img_path, save_roi=True)
            if not mrz:
                mrz = self._retry_with_rotation(img_path)
            if not mrz:
                return self._fallback_direct_easyocr(img_path)

            roi = mrz.aux["roi"]
            if roi.dtype != np.uint8:
                roi = (roi * 255).astype(np.uint8)
            roi = cv2.resize(roi, (1110, 140))

            allow = st.ascii_letters + st.digits + "< "
            code = self.reader.readtext(roi, detail=0, allowlist=allow)

            if len(code) < 2:
                return None, None, mrz

            line1 = clean_mrz_line(code[0])
            line2 = clean_mrz_line(code[1])

            if mrz.sex and len(line2) > 20:
                line2 = line2[:20] + mrz.sex + line2[21:]

            return line1, line2, mrz
        except Exception as e:
            logger.error(f"MRZ extraction error: {e}")
            return None, None, None

    def get_data(self, img_path):
        if not os.path.exists(img_path):
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)
        if not mrz:
            return None

        # 1. Build the Raw Data Dictionary
        data = {
            "surname": mrz.surname.replace("<<", " ").strip().upper() if mrz.surname else "",
            "name": mrz.names.replace("<<", " ").strip().upper() if mrz.names else "",
            "sex": get_sex(mrz.sex),
            "date_of_birth": parse_date(mrz.date_of_birth) if mrz.date_of_birth else "",
            "nationality": get_country_name(mrz.nationality),
            "passport_type": clean_string(mrz.type),
            "passport_number": clean_string(mrz.number),
            "issuing_country": get_country_name(mrz.country),
            "expiration_date": parse_date(mrz.expiration_date),
            "personal_number": clean_string(mrz.personal_number),
            "mrz_full_string": (line1 or "") + (line2 or ""),
            "source_file": os.path.basename(img_path),
            "airline_format": self.airline,
        }

        # 2. VALIDATE DATA (The "Final Fix" implementation)
        # Using the updated function signature with the 'airline' parameter
        is_valid, errors = validate_passport_data(data, self.airline)
        
        # 3. Add validation results to the dictionary
        data["is_valid"] = is_valid
        data["validation_errors"] = errors

        return data

    def process_pdf(self, pdf_path):
        os.makedirs(TEMP_DIR, exist_ok=True)
        results = []
        try:
            pages = convert_from_path(pdf_path)
        except Exception:
            import fitz
            doc = fitz.open(pdf_path)
            pages = []
            for i in range(len(doc)):
                pix = doc.load_page(i).get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
            doc.close()

        for i, page in enumerate(pages):
            img_path = os.path.join(TEMP_DIR, f"page_{i}.png")
            page.save(img_path, "PNG")
            data = self.get_data(img_path)
            if data:
                results.append(data)
            os.remove(img_path)

        return results
