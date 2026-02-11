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
