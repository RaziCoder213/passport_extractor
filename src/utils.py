import string as st
from dateutil import parser
import logging
import sys
import re
from config.settings import COUNTRY_CODES

def setup_logger(name=__name__):
    """Sets up a logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger(__name__)

def parse_date(date_obj, iob=True):
    """Parses a date object or string into DD/MM/YYYY format."""
    try:
        date_str = date_obj.isoformat() if hasattr(date_obj, 'isoformat') else str(date_obj)
        # Passport dates are often YYMMDD, but dateutil usually handles it if formatted correctly.
        # However, MRZ dates are tricky. PassportEye usually returns YYMMDD.
        # parser.parse might struggle with 2-digit years without context, but let's trust existing logic first.
        date = parser.parse(date_str, yearfirst=True).date()
        return date.strftime('%d/%m/%Y')
    except (ValueError, TypeError) as e:
        logger.debug(f"Date parsing failed for {date_obj}: {e}")
        return str(date_obj)

def clean_string(text):
    """Removes non-alphanumeric characters and converts to uppercase."""
    if not text:
        return ""
    return ''.join(i for i in text if i.isalnum()).upper()

def clean_name_field(text):
    """
    Clean name field by removing noise and extra characters.
    Only removes junk, doesn't add or replace characters.
    """
    if not text:
        return ""
    
    text = text.upper().strip()
    
    # Remove trailing repeated characters (like KKKKKK)
    if len(text) > 2:
        # Check if last 3+ characters are the same (likely noise)
        last_char = text[-1]
        if text[-3:] == last_char * 3:
            # Find where the repetition starts
            for i in range(len(text)-1, -1, -1):
                if text[i] != last_char:
                    text = text[:i+1]
                    break
    
    # Remove any non-alphabetic characters except spaces
    cleaned = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in text)
    
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def clean_mrz_line(line: str) -> str:
    """Clean MRZ line by removing noise without adding characters."""
    if not line:
        return ""
    
    line = line.upper().replace(" ", "")
    
    # Remove any characters that shouldn't be in MRZ
    # Keep only uppercase letters, digits, and '<'
    allowed = set(st.ascii_uppercase + st.digits + "<")
    line = "".join([c for c in line if c in allowed])
    
    # Don't force length - keep what we have
    return line

def get_country_name(country_code):
    """Resolves 3-letter country code to full name."""
    country_code = str(country_code).upper()
    for c in COUNTRY_CODES:
        if c['alpha-3'] == country_code:
            return c['name'].upper()
    return country_code

def get_sex(code):
    """Standardizes sex code."""
    code = str(code).upper() if code else ''
    if code in ['M', 'F']:
        return code
    if code == '0':
        return 'M' # Fallback based on existing logic
    return code

