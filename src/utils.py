# utils.py
import string as st
import logging
import sys
from dateutil import parser

# Set up logger
def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger(__name__)

def parse_date(date_obj):
    """Parse passport date (YYMMDD) or standard date strings to DD/MM/YYYY"""
    if not date_obj:
        return ""
    try:
        date_str = str(date_obj)
        date = parser.parse(date_str, yearfirst=True).date()
        return date.strftime('%d/%m/%Y')
    except Exception as e:
        logger.debug(f"Date parsing failed: {date_obj} -> {e}")
        return str(date_obj)

def clean_name_field(text):
    """Clean name field: uppercase, remove junk, keep spaces"""
    if not text:
        return ""
    text = text.upper().strip()
    # Remove non-alpha except space
    cleaned = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in text)
    # Remove multiple spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

def clean_mrz_line(line: str) -> str:
    """Clean MRZ line: keep only allowed characters and correct length"""
    if not line:
        return ""
    line = line.upper().replace(" ", "")
    allowed = set(st.ascii_uppercase + st.digits + "<")
    line = "".join([c for c in line if c in allowed])
    # Ensure TD3 length
    if len(line) < 44:
        line += "<" * (44 - len(line))
    return line[:44]

def parse_mrz_names(line1):
    """
    Extract surname and given names from MRZ line1 (TD3 format)
    """
    if not line1 or '<<' not in line1:
        return "", ""
    parts = line1.split("<<")
    surname = parts[0].replace("<", " ").strip()
    given_names = parts[1].replace("<", " ").strip()
    surname = ' '.join(surname.split())
    given_names = ' '.join(given_names.split())
    return surname, given_names
