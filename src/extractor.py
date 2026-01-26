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

# Custom Imports - Ensure these files exist in src/
from src.utils import (
    clean_string, clean_mrz_line, parse_date, 
    get_country_name, get_sex, setup_logger
)
from src.validator import validate_passport_data  # <--- CRITICAL IMPORT
from src.fallback_mrz import FallbackMRZ
from config.settings import USE_GPU, OCR_LANGUAGES, TEMP_DIR

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

class PassportExtractor:
    def __init__(self, use_gpu=USE_GPU, languages=None, airline=None):
        self.use_gpu = use_gpu
        self.languages = languages if languages else OCR_LANGUAGES
        self.airline = airline or "iraq" # Default to iraq if None

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "data", "models")
        os.makedirs(model_dir, exist_ok=True)

        self.reader = easyocr.Reader(
            self.languages, gpu=use_gpu, model_storage_directory=model_dir
        )

    def extract_mrz_from_roi(self, img_path):
        # ... (same logic as before for OCR) ...
        try:
            mrz = read_mrz(img_path, save_roi=True)
            if not mrz: return None, None, None
            # (Simplified for brevity, use your previous OCR logic here)
            return "LINE1", "LINE2", mrz
        except:
            return None, None, None

    def get_data(self, img_path):
        if not os.path.exists(img_path):
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)
        if not mrz:
            return None

        # Build data dictionary
        data = {
            "surname": mrz.surname.replace("<<", " ").strip().upper() if mrz.surname else "",
            "name": mrz.names.replace("<<", " ").strip().upper() if mrz.names else "",
            "sex": get_sex(mrz.sex),
            "date_of_birth": parse_date(mrz.date_of_birth) if mrz.date_of_birth else "",
            "nationality": get_country_name(mrz.nationality),
            "passport_number": clean_string(mrz.number),
            "issuing_country": get_country_name(mrz.country),
            "expiration_date": parse_date(mrz.expiration_date),
            "source_file": os.path.basename(img_path),
        }

        # ✅ CORRECT CALL: Pass both data and airline
        is_valid, errors = validate_passport_data(data, self.airline)
        
        data["is_valid"] = is_valid
        data["validation_errors"] = errors

        return data
