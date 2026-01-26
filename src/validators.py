import datetime

def validate_passport_data(data, airline: str):
    """
    Validates passport data based on airline-specific rules.
    1️⃣ Update: Matches the new signature (data, airline)
    """
    errors = []
    
    if not data:
        return False, ["No data extracted from image."]

    # Standardize input
    airline = airline.lower().strip() if airline else "unknown"

    # Define logic based on the "FINAL FIX" rules
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
        # Default fallback
        errors.append(f"Unknown airline format: {airline}")
        return False, errors

    # Check for missing values
    for field in required_fields:
        val = data.get(field)
        if not val or str(val).strip() == "" or val == "N/A":
            errors.append(f"Missing required field: {field}")

    # Return boolean and list of errors
    return len(errors) == 0, errors
