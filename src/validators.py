import datetime

def validate_passport_data(data, airline: str):
    """
    Validates extracted passport data based on airline-specific rules.
    Returns: (is_valid: bool, errors: list)
    """
    errors = []
    
    if not data:
        return False, ["No data extracted from passport."]

    airline = airline.lower().strip()

    # Define required fields based on the specific formats provided
    if airline == "iraqi":
        # Required for: TYPE, TITLE, FIRST NAME, LAST NAME, DOB, GENDER
        required_fields = [
            "name",
            "surname",
            "date_of_birth",
            "sex",
        ]
    elif airline in ("flydubai", "fly_dubai", "fz"):
        # Required for: Last Name, First Name, Title, PTC, Gender, DOB, Passport No, Nat, Issue Country, Expiry
        required_fields = [
            "surname",
            "name",
            "sex",
            "date_of_birth",
            "passport_number",
            "nationality",
            "issuing_country",
            "expiration_date",
        ]
    else:
        errors.append(f"Unknown airline format: {airline}")
        return False, errors

    # Check for missing or empty fields
    for field in required_fields:
        val = data.get(field)
        if not val or str(val).strip() in ["", "N/A"]:
            errors.append(f"Missing required field: {field}")

    is_valid = len(errors) == 0
    return is_valid, errors
