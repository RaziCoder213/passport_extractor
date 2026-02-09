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
    Cleans name/surname fields from MRZ.
    Handles standard separators (<<, <) and fixes common OCR errors where fillers (<) are read as 'K'.
    """
    if not text:
        return ""
    
    # Normalize to uppercase first
    text = text.upper()
    
    # 1. Replace << with space (standard MRZ separator)
    text = text.replace("<<", " ")
    
    # 2. Replace < with space (filler within name)
    text = text.replace("<", " ")
    
    # 3. Clean up junk words (often long sequences of Ks from OCR errors)
    words = text.split()
    cleaned_words = []
    
    # Heuristic: Detect if we have a tail of single/double char junk
    # Iterate backwards? No, let's just process forward but be aware of the "tail"
    
    for i, word in enumerate(words):
        # Check if word is suspicious (mostly Ks or <s)
        
        # 3a. Long junk words (handles KKKKKKKKKKEKKKK)
        if len(word) >= 3:
            k_count = word.count('K')
            ratio = k_count / len(word)
            
            # If more than 70% of the word is K, assume it's garbage and stop
            if ratio > 0.7:
                break
        
        # 3b. Sequence of short junk words (handles K K K K)
        # If this word is short (<3 chars) and mostly K, AND all subsequent words are also junk-like?
        # This is risky for "MARK K" (middle name K).
        # But if we have 3 or more junk-like tokens in a row?
        
        # Simplified: If we see 3+ consecutive tokens that are single chars 'K', strip them.
        # Check ahead
        if len(word) <= 2:
            is_junk = all(c in 'K' for c in word)
            if is_junk:
                # Look ahead to see if it's a sequence
                # We need at least 2 junk tokens to call it junk? "MARK K" -> Keep. "MARK K K" -> Strip?
                # User issue was aggressive.
                
                # Let's count how many junk tokens follow (including this one)
                junk_sequence_len = 0
                for next_word in words[i:]:
                    if len(next_word) <= 2 and all(c in 'K' for c in next_word):
                        junk_sequence_len += 1
                    else:
                        break
                
                # If we have a sequence of 3 or more junk tokens, or if it's the very last token and it's 'K' (risky?), no.
                # Let's say if we have >= 2 junk tokens, we stop here.
                if junk_sequence_len >= 2:
                    break
        
        cleaned_words.append(word)
            
    return " ".join(cleaned_words).strip()

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
