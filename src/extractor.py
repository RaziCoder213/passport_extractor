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
try:
    from pyzbar import pyzbar
    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False

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

    def clean_image_for_ocr(self, image):
        """
        Comprehensive image cleaning for better OCR results.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Apply sharpening kernel
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to clean up
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in clean_image_for_ocr: {e}")
            # Fallback to simple grayscale conversion
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                return image

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

    def find_mrz_region(self, image):
        """
        Find the MRZ region in the image using morphological operations.
        """
        try:
            # Resize image for faster processing, maintaining aspect ratio
            h, w = image.shape[:2]
            aspect_ratio = w / h
            new_h = 600
            new_w = int(new_h * aspect_ratio)
            resized = cv2.resize(image, (new_w, new_h))

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Blackhat morphological operation to reveal dark text on light background
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

            # Sobel operator to find vertical gradients
            grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            grad_x = np.absolute(grad_x)
            (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")

            # Close gaps between characters and apply threshold
            grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
            thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Close gaps between lines and perform erosions/dilations
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by aspect ratio and size to find the MRZ
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                ar = w / float(h)
                
                # Heuristic: MRZ is wide and short
                if ar > 4 and w > new_w * 0.75:
                    # Scale back to original image size
                    orig_x = int(x * (image.shape[1] / new_w))
                    orig_y = int(y * (image.shape[0] / new_h))
                    orig_w = int(w * (image.shape[1] / new_w))
                    orig_h = int(h * (image.shape[0] / new_h))
                    
                    # Add some padding
                    pX = int((orig_x + orig_w) * 0.03)
                    pY = int((orig_y + orig_h) * 0.05)
                    (x, y) = (orig_x - pX, orig_y - pY)
                    (w, h) = (orig_w + (pX * 2), orig_h + (pY * 2))
                    
                    return (x, y, w, h)
            
            return None
        except Exception as e:
            logger.error(f"Error finding MRZ region: {e}")
            return None

    def extract_from_barcode(self, image_path):
        """
        Extract passport details automatically from barcode (PDF417)
        """
        if not BARCODE_AVAILABLE:
            logger.warning("pyzbar not available, skipping barcode detection")
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Image not found: {image_path}")
                return None

            barcodes = pyzbar.decode(image)
            if not barcodes:
                logger.warning("No barcode found in passport image")
                return None

            # Assume the first PDF417 barcode is the passport data
            for barcode in barcodes:
                if barcode.type != "PDF417":
                    continue

                barcode_data = barcode.data.decode('utf-8')
                logger.info(f"Found PDF417 barcode with data: {barcode_data[:50]}...")
                
                # US / Canada / MRZ embedded usually has fields separated by newlines or fixed width
                # Example (US Passport card):
                # @\nLASTNAME<FIRSTNAME<MIDDLE<\n1234567890USA...
                lines = barcode_data.split('\n')
                if len(lines) < 2:
                    continue

                # Extract names from first line
                name_line = lines[0].replace('@','').replace('<',' ').strip()
                surname, given_names = name_line.split(' ', 1) if ' ' in name_line else (name_line, '')

                # Parse remaining fields (simplified)
                passport_number = lines[1][:9]  # first 9 digits
                nationality = lines[1][9:12]
                dob = parse_date(lines[1][12:18])
                sex = lines[1][18]
                expiry = parse_date(lines[1][19:25])

                return {
                    "surname": clean_name_field(surname),
                    "given_names": clean_name_field(given_names),
                    "passport_number": passport_number,
                    "nationality": nationality,
                    "sex": sex,
                    "date_of_birth": dob,
                    "expiration_date": expiry,
                    "barcode_data": barcode_data
                }

            return None
        except Exception as e:
            logger.error(f"Error extracting from barcode: {e}")
            return None

    def get_data(self, img_path, airline=None):
        """
        Extracts full passport data from an image file.
        Returns a dictionary of extracted fields.
        """
        if not os.path.exists(img_path):
            logger.error(f"File not found: {img_path}")
            return {"error": "File not found"}

        try:
            # Load image with OpenCV
            image = cv2.imread(img_path)
            if image is None:
                return {"error": "Could not read image"}

            # 1. Try to find MRZ region first
            mrz_coords = self.find_mrz_region(image)
            mrz_lines = []
            if mrz_coords:
                (x, y, w, h) = mrz_coords
                mrz_roi = image[y:y+h, x:x+w]
                
                # Clean the ROI for better OCR
                cleaned_roi = self.clean_image_for_ocr(mrz_roi)
                
                # Use easyocr on the cleaned ROI
                result = self.reader.readtext(cleaned_roi, detail=0, paragraph=False)
                
                if len(result) >= 2:
                    # Assume the last two lines are the MRZ
                    mrz_lines = [clean_mrz_line(line) for line in result[-2:]]
            
            # 2. If MRZ not found with our method, use PassportEye
            if not mrz_lines:
                mrz = read_mrz(img_path, save_roi=True)
                if mrz:
                    mrz_lines = [clean_mrz_line(line) for line in mrz.aux['mrz_text'].split('\n')]
                else:
                    # 3. Fallback to rotation
                    mrz = self._retry_with_rotation(img_path)
                    if mrz:
                        mrz_lines = [clean_mrz_line(line) for line in mrz.aux['mrz_text'].split('\n')]

            # 4. If still no MRZ, use direct EasyOCR fallback
            if not mrz_lines:
                line1, line2, _ = self._fallback_direct_easyocr(img_path)
                if line1 and line2:
                    mrz_lines = [line1, line2]

            # If we have MRZ lines, parse them
            if mrz_lines and len(mrz_lines) >= 2:
                line1, line2 = mrz_lines[-2], mrz_lines[-1]
                mrz_obj = FallbackMRZ(line1, line2)
                
                # Final data structure
                passport_data = {
                    "surname": clean_name_field(mrz_obj.surname),
                    "names": clean_name_field(mrz_obj.names),
                    "passport_number": clean_string(mrz_obj.number),
                    "country_code": clean_string(mrz_obj.country),
                    "nationality": get_country_name(mrz_obj.nationality),
                    "date_of_birth": parse_date(mrz_obj.date_of_birth),
                    "sex": get_sex(mrz_obj.sex),
                    "expiration_date": parse_date(mrz_obj.expiration_date),
                    "personal_number": clean_string(mrz_obj.personal_number),
                    "mrz_string": f"{line1}\n{line2}"
                }
                return passport_data

            # 5. If all MRZ methods fail, return an error
            logger.error("All MRZ detection methods failed.")
            return {"error": "Could not extract MRZ from the image."}

        except Exception as e:
            logger.error(f"An error occurred during get_data: {e}")
            return {"error": str(e)}
