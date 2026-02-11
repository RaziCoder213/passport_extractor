import cv2
from pyzbar import pyzbar
from utils import parse_date, clean_name_field, setup_logger

logger = setup_logger(__name__)

def extract_passport_from_barcode(image_path):
    """
    Extract passport details automatically from barcode (PDF417)
    """
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

# Example usage
if __name__ == "__main__":
    data = extract_passport_from_barcode("passport.jpg")
    print(data)
