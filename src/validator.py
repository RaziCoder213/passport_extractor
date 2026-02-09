import datetime

def validate_passport_data(data):
    """
    Validates extracted passport data.
    Returns a list of validation warnings/errors.
    """
    errors = []
    
    if not data:
        return ["No data to validate"]

    # Check required fields
    required_fields = ['surname', 'name', 'passport_number', 'nationality']
    for field in required_fields:
        if not data.get(field):
            errors.append(f"Missing required field: {field}")

    # Validate dates (basic check if they look like DD/MM/YYYY)
    date_fields = ['date_of_birth', 'expiration_date']
    for field in date_fields:
        val = data.get(field)
        if val:
            try:
                datetime.datetime.strptime(val, '%d/%m/%Y')
            except ValueError:
                errors.append(f"Invalid date format for {field}: {val} (expected DD/MM/YYYY)")

    # Validate MRZ length roughly (should be around 88 chars for TD3)
    mrz = data.get('mrz_full_string', '')
    if len(mrz) < 80:
        errors.append(f"MRZ string seems too short ({len(mrz)} chars)")

    return errors
