import os
import cv2
import numpy as np
import easyocr
import warnings
import re
from passporteye import read_mrz
from src.utils import clean_string, clean_mrz_line, parse_date, get_country_name, get_sex, setup_logger
from src.validator import validate_passport_data

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

class PassportExtractor:
    def __init__(self, use_gpu=False, languages=None, airline="iraqi"):
        self.use_gpu = use_gpu
        self.languages = languages if languages else ["en"]
        self.airline = airline.lower().strip()
        self.reader = easyocr.Reader(self.languages, gpu=use_gpu)

    def clean_noise(self, text):
        """Removes stray 'K' characters often picked up by OCR at the end of lines"""
        if not text: return ""
        cleaned = re.sub(r'\s+K\s+K*$', '', text)
        cleaned = re.sub(r'\s+K+$', '', cleaned)
        return cleaned.strip().upper()

    def extract_mrz_from_roi(self, img_path):
        try:
            mrz = read_mrz(img_path)
            if not mrz: return None, None, None
            roi = mrz.aux.get("roi")
            if roi is not None:
                if roi.dtype != np.uint8: roi = (roi * 255).astype(np.uint8)
                code = self.reader.readtext(roi, detail=0)
                if len(code) >= 2:
                    return clean_mrz_line(code[0]), clean_mrz_line(code[1]), mrz
            return None, None, mrz
        except:
            return None, None, None

    def get_data(self, img_path):
        if not os.path.exists(img_path): return None
        line1, line2, mrz = self.extract_mrz_from_roi(img_path)
        if not mrz: return None

        raw = {
            "surname": self.clean_noise(mrz.surname.replace("<<", " ") if mrz.surname else ""),
            "name": self.clean_noise(mrz.names.replace("<<", " ") if mrz.names else ""),
            "sex": get_sex(mrz.sex), # Expected 'M' or 'F'
            "date_of_birth": parse_date(mrz.date_of_birth) or "",
            "nationality": get_country_name(mrz.nationality),
            "passport_number": clean_string(mrz.number),
            "issuing_country": get_country_name(mrz.country),
            "expiration_date": parse_date(mrz.expiration_date) or "",
        }

        # Dynamic Title Logic
        title_val = "MR" if raw["sex"] == "M" else "MRS"

        if self.airline == "iraqi":
            return {
                "TYPE": "Adult",
                "TITLE": title_val,  # Fixed: MR or MRS only
                "FIRST NAME": raw["name"],
                "LAST NAME": raw["surname"],
                "DOB (DD/MM/YYYY)": raw["date_of_birth"],
                "GENDER": "Male" if raw["sex"] == "M" else "Female"
            }
        else: # flydubai
            return {
                "Last Name": raw["surname"],
                "First Name and Middle Name": raw["name"],
                "Title": title_val, # Fixed: MR or MRS only
                "PTC": "ADT",
                "Gender": raw["sex"],
                "Date of Birth": raw["date_of_birth"],
                "Passport Number": raw["passport_number"],
                "Passport Nationality": raw["nationality"],
                "Passport Issue Country": raw["issuing_country"],
                "Passport Expiry Date": raw["expiration_date"]
            }
