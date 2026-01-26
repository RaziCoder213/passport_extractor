import datetime

def validate_passport_data(data, airline: str):
    """
    Validates required fields based on airline rules.
    Returns: (is_valid: bool, errors: list)
    """
    errors = []
    if not data:
        return False, ["No data extracted."]

    airline = airline.lower().strip()

    if airline == "iraqi":
        required_fields = ["name", "surname", "date_of_birth", "sex"]
    elif airline in ("flydubai", "fly_dubai", "fz"):
        required_fields = [
            "surname", "name", "sex", "date_of_birth", 
            "passport_number", "nationality", "issuing_country", "expiration_date"
        ]
    else:
        return False, [f"Unknown airline: {airline}"]

    for field in required_fields:
        val = data.get(field)
        if not val or str(val).strip() in ["", "N/A"]:
            errors.append(f"Missing required field: {field}")

    return len(errors) == 0, errors
