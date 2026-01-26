import datetime

def validate_passport_data(data, airline: str):
    """
    Validates extracted passport data based on airline-specific rules.
    1. Iraq: Requires Sex, does not strictly require Issuing Country.
    2. FlyDubai: Requires Issuing Country, does not strictly require Sex.
    """
    errors = []
    
    if not data:
        return False, ["No data to validate"]

    # Normalize airline input
    airline = airline.lower().strip() if airline else "unknown"

    # 1. Define required fields based on airline rules
    if airline == "iraq":
        required_fields = [
            "surname",
            "name",
            "passport_number",
            "date_of_birth",
            "expiration_date",
            "nationality",
            "sex",
        ]
    elif airline in ("flydubai", "fly_dubai", "fz"):
        required_fields = [
            "surname",
            "name",
            "passport_number",
            "date_of_birth",
            "expiration_date",
            "issuing_country",
            "nationality",
        ]
    else:
        # If the airline is not recognized
        errors.append(f"Unknown airline format: {airline}")
        return False, errors

    # 2. Check for missing or empty required fields
    for field in required_fields:
        if not data.get(field) or str(data.get(field)).strip() == "":
            errors.append(f"Missing required field: {field}")

    # 3. Validate Date Formats (Optional but recommended)
    date_fields = ['date_of_birth', 'expiration_date']
    for field in date_fields:
        val = data.get(field)
        if val and val != "N/A":
            try:
                # Assuming your parse_date util returns DD/MM/YYYY
                if "/" in val:
                    datetime.datetime.strptime(val, '%d/%m/%Y')
            except ValueError:
                errors.append(f"Invalid date format for {field}: {val}")

    # Return boolean status and the list of errors
    is_valid = len(errors) == 0
    return is_valid, errors
