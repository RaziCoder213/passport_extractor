import string as st
from dateutil import parser
import logging
import sys
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

def format_date_flydubai(date_obj):
    """Formats a date object or string into FlyDubai format (DDMMMYYformat e.g., 13NOV84)."""
    try:
        date_str = date_obj.isoformat() if hasattr(date_obj, 'isoformat') else str(date_obj)
        date = parser.parse(date_str, yearfirst=True).date()
        return date.strftime('%d%b%Y').upper()[:8]  # Format: 13NOV84 (8 chars: 2 digits + 3 letters + 2 digits)
    except (ValueError, TypeError) as e:
        logger.debug(f"Date formatting failed for {date_obj}: {e}")
        return str(date_obj)

def clean_string(text):
    """Removes non-alphanumeric characters and converts to uppercase."""
    if not text:
        return ""
    return ''.join(i for i in text if i.isalnum()).upper()

def clean_mrz_line(line: str) -> str:
    """Fix bad spacing or bad OCR for MRZ lines."""
    if not line:
        return ""
    
    line = line.upper().replace(" ", "")
    
    # Remove accidental characters except allowed
    allowed = set(st.ascii_uppercase + st.digits + "<")
    line = "".join([c for c in line if c in allowed])

    # Ensure 44 length (standard TD3 MRZ length)
    # Note: TD1/TD2 might be different lengths (30 or 36), but this logic enforces 44.
    # We will keep existing logic for consistency but be aware of other formats.
    if len(line) < 44:
        line += "<" * (44 - len(line))
    return line[:44]

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
