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

# Fix for SSL certificate errors on Mac when downloading EasyOCR models
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

# Suppress warnings
warnings.filterwarnings('ignore')

logger = setup_logger(__name__)

class PassportExtractor:
    def __init__(self, use_gpu=USE_GPU, languages=None):
        self.languages = languages if languages else OCR_LANGUAGES
        
        # Set model storage directory to project/data/models to avoid permission issues
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'data', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Initializing EasyOCR Reader (GPU={use_gpu})...")
        self.reader = easyocr.Reader(self.languages, gpu=use_gpu, model_storage_directory=model_dir)
        logger.info("EasyOCR Reader initialized.")

    def _retry_with_rotation(self, img_path):
        """Try rotating image 90, 180, 270 degrees to find MRZ."""
        try:
            original = Image.open(img_path)
            
            for angle in [90, 180, 270]:
                logger.info(f"Retrying with rotation: {angle} degrees")
                # expand=True ensures the whole image is kept
                rotated = original.rotate(angle, expand=True)
                
                # Save to a temp file
                temp_rot_path = img_path + f"_rot_{angle}.png"
                rotated.save(temp_rot_path)
                
                mrz = read_mrz(temp_rot_path, save_roi=True)
                
                # Cleanup
                if os.path.exists(temp_rot_path):
                    os.remove(temp_rot_path)
                
                if mrz:
                    logger.info(f"MRZ detected after rotation {angle}")
                    return mrz
                    
            return None
        except Exception as e:
            logger.error(f"Rotation fallback failed: {e}")
            return None

    def _fallback_direct_easyocr(self, img_path):
        """
        Fallback method: Read the entire image with EasyOCR and try to find MRZ lines.
        """
        try:
            # Read full image
            # detail=0 returns just the list of strings
            result = self.reader.readtext(img_path, detail=0)
            
            # Filter and clean lines
            potential_lines = []
            for line in result:
                clean = clean_mrz_line(line)
                # Heuristic: MRZ lines are usually long (30-44 chars) and contain '<<' or start with P<, I<
                if len(clean) > 30 and ('<<' in clean or clean.startswith(('P<', 'I<', 'A<', 'V<'))):
                    potential_lines.append(clean)
            
            # Look for the last two valid lines (TD3 format usually has 2 lines at the bottom)
            if len(potential_lines) >= 2:
                # Assume the last two are the MRZ
                line1 = potential_lines[-2]
                line2 = potential_lines[-1]
                
                # Basic validation: Line 1 usually starts with P, I, A, V
                if not line1[0] in 'PIAV':
                     # Maybe we picked wrong lines. Let's look for a line starting with P/I/A/V
                     for i, l in enumerate(potential_lines):
                         if l.startswith(('P<', 'I<', 'A<', 'V<')) and i+1 < len(potential_lines):
                             line1 = l
                             line2 = potential_lines[i+1]
                             break
                
                logger.info(f"Direct EasyOCR found potential MRZ: {line1} / {line2}")
                mrz_obj = FallbackMRZ(line1, line2)
                return line1, line2, mrz_obj
            
            return None, None, None

        except Exception as e:
            logger.error(f"Direct EasyOCR fallback failed: {e}")
            return None, None, None

    def extract_mrz_from_roi(self, img_path):
        """
        Extracts MRZ lines using PassportEye to find ROI, then EasyOCR to read text.
        Returns (line1, line2, mrz_object).
        """
        try:
            # Analyze image with PassportEye
            mrz = read_mrz(img_path, save_roi=True)
            
            if not mrz:
                logger.warning(f"PassportEye failed to detect MRZ in {img_path}. Trying rotations...")
                # Try rotating 90, 180, 270
                mrz = self._retry_with_rotation(img_path)
            
            if not mrz:
                logger.warning(f"PassportEye failed to detect MRZ in {img_path} after rotations.")
                # Fallback 2: Direct EasyOCR on the full image
                logger.info("Attempting Direct EasyOCR fallback on full image...")
                return self._fallback_direct_easyocr(img_path)

            # Get ROI (Region of Interest)
            roi = mrz.aux['roi']
            
            # Ensure ROI is uint8 for OpenCV
            if roi.dtype != np.uint8:
                if roi.max() <= 1.0:
                    roi = (roi * 255).astype(np.uint8)
                else:
                    roi = roi.astype(np.uint8)

            # Preprocess ROI for EasyOCR
            # Original code: saved to tmp.png (gray), read back, resized to (1110, 140)
            # We will try to do this in memory using OpenCV
            
            # roi is likely a numpy array (H, W) or (H, W, C). PassportEye usually returns grayscale for ROI?
            # Let's normalize to BGR for OpenCV consistency if needed, or keep grayscale.
            
            # Resize to improve OCR accuracy as per original code logic
            # Note: (1110, 140) is the target size (Width, Height)
            img_resized = cv2.resize(roi, (1110, 140))
            
            # Define allowed characters for MRZ
            allow = st.ascii_letters + st.digits + "<"
            
            # Run EasyOCR
            code = self.reader.readtext(img_resized, detail=0, allowlist=allow)

            if len(code) < 2:
                logger.warning(f"EasyOCR found fewer than 2 lines in ROI for {img_path}")
                return None, None, mrz

            line1 = clean_mrz_line(code[0])
            line2 = clean_mrz_line(code[1])

            # Correct sex at index 20 of line 2 if available from mrz object
            # MRZ object might have parsed it correctly even if EasyOCR missed it
            if mrz.sex and len(line2) > 20:
                l2_list = list(line2)
                l2_list[20] = mrz.sex
                line2 = "".join(l2_list)

            return line1, line2, mrz

        except Exception as e:
            logger.error(f"Error in extract_mrz_from_roi: {e}")
            return None, None, None

    def get_data(self, img_path, airline=None):
        """
        Extracts full passport data from an image file.
        Returns a dictionary of extracted fields.
        """
        if not os.path.exists(img_path):
            logger.error(f"File not found: {img_path}")
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)

        if mrz is None:
            if line1 and line2:
                mrz = FallbackMRZ(line1, line2)

        if mrz is None:
            logger.warning(f"Could not extract a valid MRZ from {img_path}")
            return None
        
        # Safely get data from MRZ object
        mrz_data = mrz.to_dict() if hasattr(mrz, 'to_dict') else {}
        
        surname = clean_name_field(mrz_data.get('surname', ''))
        name = clean_name_field(mrz_data.get('name', ''))

        data = {
            "surname": surname,
            "name": name,
            "country": get_country_name(mrz_data.get('country', '')),
            "nationality": get_country_name(mrz_data.get('nationality', '')),
            "passport_number": clean_string(mrz_data.get('number', '')),
            "sex": get_sex(mrz_data.get('sex', '')),
            "date_of_birth": parse_date(mrz_data.get('date_of_birth', '')),
            "expiration_date": parse_date(mrz_data.get('expiration_date', '')),
            "mrz_line1": line1,
            "mrz_line2": line2,
            "valid_score": mrz_data.get('valid_score', 0),
        }
        return data

    def process_pdf(self, pdf_path, airline=None):
        """
        Extracts passport data from all pages of a PDF file.
        Returns a list of dictionaries, one for each page with a valid MRZ.
        """
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        try:
            images = convert_from_path(pdf_path, dpi=300)
            results = []
            for i, img in enumerate(images):
                page_path = os.path.join(TEMP_DIR, f"page_{i}.png")
                img.save(page_path, 'PNG')
                
                data = self.get_data(page_path, airline=airline)
                if data:
                    data['source_page'] = i + 1
                    results.append(data)
            
            return results
        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {e}")
            return []

        data = {}
        # Use PassportEye's parsing where possible, fallback/clean as needed
        data['surname'] = clean_name_field(mrz.surname)
        data['name'] = clean_name_field(mrz.names)
        data['sex'] = get_sex(mrz.sex)
        data['date_of_birth'] = parse_date(mrz.date_of_birth) if mrz.date_of_birth else ""
        data['nationality'] = get_country_name(mrz.nationality)
        data['passport_type'] = clean_string(mrz.type)
        data['passport_number'] = clean_string(mrz.number)
        data['issuing_country'] = get_country_name(mrz.country)
        data['expiration_date'] = parse_date(mrz.expiration_date)
        data['personal_number'] = clean_string(mrz.personal_number)
        
        # Construct full MRZ string
        # Prefer the OCR'd lines if they exist, otherwise fallback?
        # The original code returned (line1 or "") + (line2 or "")
        data['mrz_full_string'] = (line1 or "") + (line2 or "")
        data['source_file'] = os.path.basename(img_path)

        return data

    def process_pdf(self, pdf_path):
        """
        Converts PDF pages to images and extracts data from each.
        Returns a list of data dictionaries.
        """
        extracted_data = []
        
        # Ensure temp dir exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        logger.info(f"Processing PDF: {pdf_path}")
        try:
            pages = convert_from_path(pdf_path)
        except Exception as e:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                pages = []
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pages.append(img)
                doc.close()
            except Exception as e2:
                logger.error(f"Failed to convert PDF {pdf_path}: {e2}")
                return []

        for i, page in enumerate(pages):
            temp_img_path = os.path.join(TEMP_DIR, f"temp_page_{i+1}.png")
            page.save(temp_img_path, "PNG")
            
            logger.info(f"Processing page {i+1}...")
            result = self.get_data(temp_img_path)
            
            # Special MRZ fix for PDF from original code
            # It seems to re-clean the combined string.
            if result and result.get("mrz_full_string"):
                full_mrz = result["mrz_full_string"]
                if len(full_mrz) >= 88: # 44 * 2
                    l1 = full_mrz[:44]
                    l2 = full_mrz[44:]
                    l1 = clean_mrz_line(l1)
                    l2 = clean_mrz_line(l2)
                    result["mrz_full_string"] = l1 + l2
            
            if result:
                extracted_data.append(result)
            
            # Cleanup temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        return extracted_data
