import os
import cv2
import numpy as np
import easyocr
import warnings
import ssl
from passporteye import read_mrz
from PIL import Image
import string as st

# Internal imports
from src.utils import (
    clean_string, clean_mrz_line, parse_date,
    get_country_name, get_sex, setup_logger
)
from src.validator import validate_passport_data
from src.fallback_mrz import FallbackMRZ
from config.settings import USE_GPU, OCR_LANGUAGES, TEMP_DIR

# Fix SSL issues for model downloading
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

class PassportExtractor:
    def __init__(self, use_gpu=USE_GPU, languages=None, airline=None):
        self.use_gpu = use_gpu
        self.languages = languages if languages else OCR_LANGUAGES
        self.airline = airline or "iraqi"

        # Initialize EasyOCR
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "data", "models")
        os.makedirs(model_dir, exist_ok=True)

        self.reader = easyocr.Reader(
            self.languages, gpu=use_gpu, model_storage_directory=model_dir
        )

    def extract_mrz_from_roi(self, img_path):
        """Attempts to find and read the MRZ region."""
        try:
            mrz = read_mrz(img_path, save_roi=True)
            if not mrz:
                return None, None, None
            
            roi = mrz.aux["roi"]
            if roi.dtype != np.uint8:
                roi = (roi * 255).astype(np.uint8)
            roi = cv2.resize(roi, (1110, 140))
            
            allow = st.ascii_letters + st.digits + "< "
            code = self.reader.readtext(roi, detail=0, allowlist=allow)
            
            if len(code) < 2:
                return None, None, mrz
            
            return clean_mrz_line(code[0]), clean_mrz_line(code[1]), mrz
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return None, None, None

    def get_data(self, img_path):
        """Extracts and formats data based on airline-specific requirements."""
        if not os.path.exists(img_path):
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)
        if not mrz:
            return None

        # 1. Raw Extraction
        raw_data = {
            "surname": mrz.surname.replace("<<", " ").strip().upper() if mrz.surname else "",
            "name": mrz.names.replace("<<", " ").strip().upper() if mrz.names else "",
            "sex": get_sex(mrz.sex), # Returns 'M', 'F', or 'X'
            "date_of_birth": parse_date(mrz.date_of_birth) if mrz.date_of_birth else "",
            "nationality": get_country_name(mrz.nationality),
            "passport_number": clean_string(mrz.number),
            "issuing_country": get_country_name(mrz.country),
            "expiration_date": parse_date(mrz.expiration_date),
            "mrz_full_string": (line1 or "") + (line2 or ""),
        }

        # 2. Run Validation
        is_valid, errors = validate_passport_data(raw_data, self.airline)
        val_status = "OK" if is_valid else "; ".join(errors)

        # 3. Apply Final Airline Formatting
        airline_key = self.airline.lower().strip()

        if airline_key == "iraqi":
            return {
                "TYPE": "Adult",
                "TITLE": "MR" if raw_data["sex"] == "M" else "MRS",
                "FIRST NAME": raw_data["name"],
                "LAST NAME": raw_data["surname"],
                "DOB (DD/MM/YYYY)": raw_data["date_of_birth"],
                "GENDER": "Male" if raw_data["sex"] == "M" else "Female",
                "Validation": val_status
            }
        
        elif airline_key in ("flydubai", "fly_dubai", "fz"):
            return {
                "Last Name": raw_data["surname"],
                "First Name and Middle Name": raw_data["name"],
                "Title": "MR" if raw_data["sex"] == "M" else "MRS",
                "PTC": "ADT",
                "Gender": raw_data["sex"], # Returns M/F
                "Date of Birth": raw_data["date_of_birth"],
                "Passport Number": raw_data["passport_number"],
                "Passport Nationality": raw_data["nationality"],
                "Passport Issue Country": raw_data["issuing_country"],
                "Passport Expiry Date": raw_data["expiration_date"],
                "Validation": val_status
            }

        return raw_data # Fallback to raw if unknown
