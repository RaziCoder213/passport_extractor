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
            allow = st.ascii_uppercase + st.digits + "<"
            
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

    def preprocess_for_ocr(self, img):
        """
        Preprocess image for better OCR accuracy
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding for better text contrast
            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return clean
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return img

    def correct_common_ocr_errors(self, text):
        """
        Correct common OCR errors in passport text
        """
        if not text:
            return text
        
        # Common OCR substitutions in passport context
        corrections = {
            '0': 'O',  # Zero to letter O
            '1': 'I',  # One to letter I
            '5': 'S',  # Five to letter S
            '8': 'B',  # Eight to letter B
            '2': 'Z',  # Two to letter Z (in some contexts)
            '6': 'G',  # Six to letter G
            '7': 'T',  # Seven to letter T
        }
        
        # Apply corrections only for single characters that are clearly errors
        corrected_text = ""
        for char in text:
            if char in corrections and len(text) > 1:
                # Only correct if it makes sense in context (e.g., not in numbers)
                if not text.isdigit():
                    corrected_text += corrections[char]
                else:
                    corrected_text += char
            else:
                corrected_text += char
        
        return corrected_text
    
    def correct_name_patterns(self, name):
        """
        Apply pattern-based corrections for common name OCR errors
        """
        if not name:
            return name
        
        # Common name pattern corrections
        name = re.sub(r'\bAHMED\b', 'AHMED', name, flags=re.IGNORECASE)
        name = re.sub(r'\bMOHAMMED\b', 'MOHAMMED', name, flags=re.IGNORECASE)
        name = re.sub(r'\bMOHAMED\b', 'MOHAMED', name, flags=re.IGNORECASE)
        name = re.sub(r'\bMUHAMMAD\b', 'MUHAMMAD', name, flags=re.IGNORECASE)
        
        # Remove multiple spaces
        name = re.sub(r'\s+', ' ', name)
        
        # Ensure proper spacing around initials
        name = re.sub(r'([A-Z])\.?([A-Z])', r'\1 \2', name)
        
        return name.strip()

    def detect_text_regions(self, img):
        """
        Detect text regions in passport image using EAST text detector
        """
        try:
            # Load EAST text detector model (you'll need to download frozen_east_text_detection.pb)
            # For now, we'll use a simpler approach with contour detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to enhance text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Apply threshold
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and size (text regions are typically wide and short)
                if w > 50 and h > 10 and w/h > 2:
                    regions.append((x, y, w, h))
            
            return regions
        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")
            return []
    
    def extract_name_region(self, img, regions):
        """
        Extract the name region from detected text regions
        """
        try:
            # Sort regions by y-coordinate (top to bottom)
            regions.sort(key=lambda r: r[1])
            
            # Look for regions that might contain name fields
            name_regions = []
            for i, (x, y, w, h) in enumerate(regions):
                # Look for regions in the upper part of passport (where names typically are)
                if y < img.shape[0] // 3:
                    name_regions.append((x, y, w, h))
            
            if name_regions:
                # Take the first few regions as potential name areas
                x_min = min(r[0] for r in name_regions)
                y_min = min(r[1] for r in name_regions)
                x_max = max(r[0] + r[2] for r in name_regions)
                y_max = max(r[1] + r[3] for r in name_regions)
                
                # Add some padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(img.shape[1], x_max + padding)
                y_max = min(img.shape[0], y_max + padding)
                
                return img[y_min:y_max, x_min:x_max]
            
            return None
        except Exception as e:
            logger.warning(f"Name region extraction failed: {e}")
            return None

    def extract_given_name_from_visual_zone(self, img_path):
        """
        Extracts the given name from the visual inspection zone of the passport
        (not from MRZ). Looks for 'Given Name' or 'Given Names' field.
        """
        try:
            # Load the original image
            original_img = cv2.imread(img_path)
            if original_img is None:
                logger.error(f"Failed to load image: {img_path}")
                return None
            
            # Detect text regions in the image
            text_regions = self.detect_text_regions(original_img)
            
            # Extract name-specific region for better OCR focus
            name_region = self.extract_name_region(original_img, text_regions)
            
            # Preprocess image for better OCR accuracy
            if name_region is not None:
                # Use the name-specific region if detected
                processed_img = self.preprocess_for_ocr(name_region)
                logger.info("Using name-specific region for OCR")
            else:
                # Fallback to full image preprocessing
                processed_img = self.preprocess_for_ocr(original_img)
                logger.info("Using full image for OCR")
            
            # Save processed image temporarily for OCR
            temp_processed_path = img_path.replace('.jpg', '_processed.jpg').replace('.png', '_processed.png')
            cv2.imwrite(temp_processed_path, processed_img)
            
            # Read the processed image with EasyOCR
            results = self.reader.readtext(temp_processed_path)
            
            # Cleanup temp file
            if os.path.exists(temp_processed_path):
                os.remove(temp_processed_path)
            
            # Convert results to text lines for easier processing
            text_lines = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence readings
                    text_lines.append(text.upper().strip())
            
            # Look for given name field
            given_name = None
            for i, line in enumerate(text_lines):
                # Look for "Given Name" or "Given Names" field using regex for better matching
                if re.search(r'GIVEN\s+NAME', line, re.IGNORECASE):
                    # Check if name is on the same line with colon separator
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1 and parts[1].strip():
                            given_name = parts[1].strip()
                        else:
                            # Name might be on next line
                            if i + 1 < len(text_lines):
                                given_name = text_lines[i + 1]
                    elif '-' in line:
                        # Handle dash separator (e.g., "GIVEN NAME - JOHN")
                        parts = line.split('-', 1)
                        if len(parts) > 1 and parts[1].strip():
                            given_name = parts[1].strip()
                        else:
                            # Name might be on next line
                            if i + 1 < len(text_lines):
                                given_name = text_lines[i + 1]
                    elif re.search(r'GIVEN\s+NAME\s*$', line, re.IGNORECASE):
                        # Name might be on next line
                        if i + 1 < len(text_lines):
                            given_name = text_lines[i + 1]
                    else:
                        # Try to extract from the same line using regex
                        match = re.search(r'GIVEN\s+NAME\s*[:\-\s]*([A-Z\s]+)', line, re.IGNORECASE)
                        if match:
                            given_name = match.group(1).strip()
                        else:
                            # If no name found after splitting, try next line
                            if i + 1 < len(text_lines):
                                given_name = text_lines[i + 1]
                    break
                # Also check for "FIRST NAME" field
                elif re.search(r'FIRST\s+NAME', line, re.IGNORECASE):
                    # Similar logic as above for FIRST NAME
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1 and parts[1].strip():
                            given_name = parts[1].strip()
                        else:
                            if i + 1 < len(text_lines):
                                given_name = text_lines[i + 1]
                    elif '-' in line:
                        parts = line.split('-', 1)
                        if len(parts) > 1 and parts[1].strip():
                            given_name = parts[1].strip()
                        else:
                            if i + 1 < len(text_lines):
                                given_name = text_lines[i + 1]
                    elif re.search(r'FIRST\s+NAME\s*$', line, re.IGNORECASE):
                        if i + 1 < len(text_lines):
                            given_name = text_lines[i + 1]
                    else:
                        match = re.search(r'FIRST\s+NAME\s*[:\-\s]*([A-Z\s]+)', line, re.IGNORECASE)
                        if match:
                            given_name = match.group(1).strip()
                        else:
                            if i + 1 < len(text_lines):
                                given_name = text_lines[i + 1]
                    break
            
            if given_name:
                # Clean the extracted name
                given_name = given_name.strip()
                # Remove common field labels that might have been captured
                given_name = re.sub(r'^(NAME|NAMES|SURNAME|GIVEN)\s*[:\-]?\s*', '', given_name, flags=re.IGNORECASE)
                # Remove any remaining common prefixes
                given_name = re.sub(r'^(SURNAME|FAMILY NAME|LAST NAME|FIRST NAME)\s*[:\-]?\s*', '', given_name, flags=re.IGNORECASE)
                # Remove any trailing field labels
                given_name = re.sub(r'\s+(SURNAME|FAMILY NAME|LAST NAME|FIRST NAME|DATE|PLACE|NATIONALITY|SEX|PASSPORT)$', '', given_name, flags=re.IGNORECASE)
                
                # Remove trailing single-letter OCR artifacts (common K/X/Z/Q)
                given_name = re.sub(r'\s*([KXZQ])$', '', given_name)
                
                # If the name is too long or contains unusual characters, it might be a false positive
                if len(given_name) > 50 or any(char.isdigit() for char in given_name):
                    return None
                
                # Apply OCR error correction
                given_name = self.correct_common_ocr_errors(given_name)
                
                # Apply pattern-based corrections
                given_name = self.correct_name_patterns(given_name)
                
                # Final validation
                if len(given_name) < 2:
                    return None
                
                logger.info(f"Extracted given name from visual zone: {given_name}")
                    
                return given_name
                
        except Exception as e:
            logger.error(f"Error extracting given name from visual zone: {e}")
        
        return None

    def get_data(self, img_path, airline=None):
        """
        Extracts full passport data from an image file.
        Returns a dictionary of extracted fields.
        """
        if not os.path.exists(img_path):
            logger.error(f"File not found: {img_path}")
            return None

        line1, line2, mrz = self.extract_mrz_from_roi(img_path)

        # If we have clean lines from EasyOCR, use them to create a new MRZ object
        # to ensure data consistency, overriding any potentially faulty data from passporteye
        if line1 and line2:
            mrz = FallbackMRZ(line1, line2)

        if mrz is None:
            logger.warning(f"Could not extract a valid MRZ from {img_path}")
            return None
        
        # ALWAYS try to extract given name from visual inspection zone first
        visual_given_name = self.extract_given_name_from_visual_zone(img_path)
        
        # Safely get data from MRZ object
        # FallbackMRZ has 'names', passporteye has 'name'. Let's check for both.
        surname = clean_name_field(getattr(mrz, 'surname', ''))
        
        # Use visual given name if available and valid, otherwise fall back to MRZ
        if visual_given_name and len(visual_given_name) > 1:
            name = clean_name_field(visual_given_name)
            logger.info(f"Using GIVEN NAME from visual zone: {name}")
        else:
            # Fallback to MRZ only if visual field not detected
            name = clean_name_field(getattr(mrz, 'names', getattr(mrz, 'name', '')))
            logger.warning("Visual given name not found, using MRZ name fallback.")

        data = {
            "surname": surname,
            "name": name,
            "country": get_country_name(getattr(mrz, 'country', '')),
            "nationality": get_country_name(getattr(mrz, 'nationality', '')),
            "passport_number": clean_string(getattr(mrz, 'number', '')),
            "sex": get_sex(getattr(mrz, 'sex', '')),
            "date_of_birth": parse_date(getattr(mrz, 'date_of_birth', '')),
            "expiration_date": parse_date(getattr(mrz, 'expiration_date', '')),
            "mrz_full_string": (line1 or "") + (line2 or ""),
            "valid_score": getattr(mrz, 'valid_score', 0),
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
