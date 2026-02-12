import pandas as pd
import os
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

def calculate_passenger_type(dob_str):
    """
    Calculate passenger type based on age from date of birth.
    
    Age groups:
    - Infants: 0-1 year (0-11 months)
    - Children: 1-17 years
    - Adults: 18+ years
    
    Returns:
        For Iraqi Airways: "Infant", "Child", "Adult"
        For Flydubai: "INF", "CHD", "ADT"
    """
    if not dob_str:
        return "Adult", "ADT"  # Default to adult if no DOB
    
    try:
        # Parse the date (assuming DD/MM/YYYY format)
        dob = datetime.strptime(dob_str, '%d/%m/%Y').date()
        today = date.today()
        
        # Calculate age in years
        age_years = today.year - dob.year
        if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
            age_years -= 1
        
        # Calculate age in months for more accurate infant check
        age_months = (today.year - dob.year) * 12 + (today.month - dob.month)
        if today.day < dob.day:
            age_months -= 1
        
        # Determine passenger type based on age
        if age_months < 12:  # Less than 12 months - Infant
            return "Infant", "INF"
        elif age_years < 18:  # 1-17 years - Child
            return "Child", "CHD"
        else:  # 18+ years - Adult
            return "Adult", "ADT"
            
    except (ValueError, TypeError):
        return "Adult", "ADT"  # Default to adult if date parsing fails

def format_iraqi_airways(data_list):
    """
    Formats data for Iraqi Airways template.
    Columns: TYPE, TITLE, FIRST NAME, LAST NAME, DOB (DD/MM/YYYY), GENDER
    Example: Adult MR FirstNameOne LastNameOne 13/8/2015 Male
    """
    formatted_rows = []
    
    for item in data_list:
        sex = item.get('sex', '').upper()
        # Map Gender to Male/Female
        gender_full = "Male" if sex == 'M' else "Female"
        title = "MR" if sex == 'M' else "MRS"
        
        # Calculate passenger type based on age
        raw_dob = item.get('date_of_birth', '')
        
        # Convert MRZ date (YYMMDD) to DD/MM/YYYY for age calculation
        ddmmyyyy_dob = ''
        if raw_dob and len(raw_dob) == 6:  # YYMMDD format
            try:
                from datetime import datetime
                mrz_date = datetime.strptime(raw_dob, '%y%m%d').date()
                ddmmyyyy_dob = mrz_date.strftime('%d/%m/%Y')
            except:
                ddmmyyyy_dob = raw_dob
        else:
            ddmmyyyy_dob = raw_dob
            
        passenger_type, _ = calculate_passenger_type(ddmmyyyy_dob)
        
        row = {
            "TYPE": passenger_type,
            "TITLE": title,
            "FIRST NAME": item.get('name', ''),
            "LAST NAME": item.get('surname', ''),
            "DOB (DD/MM/YYYY)": ddmmyyyy_dob,
            "GENDER": gender_full
        }
        formatted_rows.append(row)
        
    return pd.DataFrame(formatted_rows)

from datetime import datetime

def _to_ddmmmyy(date_str):
    """Converts DD/MM/YYYY to DDMMMYY format, e.g., 13NOV84."""
    if not date_str:
        return ""
    try:
        # Parse from DD/MM/YYYY
        dt_obj = datetime.strptime(date_str, '%d/%m/%Y')
        # Format to DDMMMYY with strict zero padding for day
        # %d should do it, but to be safe we can use string formatting
        day = dt_obj.strftime('%d')
        month_year = dt_obj.strftime('%b%y').upper()
        return f"{day}{month_year}"
    except ValueError:
        return date_str # Return original if parsing fails

def format_flydubai(data_list):
    """
    Formats data for Flydubai template.
    - Date of Birth: DDMMMYY (e.g., 13NOV84)
    - Passport Expiry Date: DDMMMYY (e.g., 13NOV84)
    """
    formatted_rows = []
    
    for item in data_list:
        sex = item.get('sex', '').upper()
        title = "MR" if sex == 'M' else "MRS" 
        
        first_middle = item.get('name', '')
        
        # Get raw date data for age calculation (in DD/MM/YYYY format)
        raw_dob = item.get('date_of_birth', '')
        raw_expiry = item.get('expiration_date', '')
        
        # Calculate passenger type based on age using DD/MM/YYYY format
        # First convert MRZ date (YYMMDD) to DD/MM/YYYY for age calculation
        ddmmyyyy_dob = ''
        if raw_dob and len(raw_dob) == 6:  # YYMMDD format
            try:
                from datetime import datetime
                mrz_date = datetime.strptime(raw_dob, '%y%m%d').date()
                ddmmyyyy_dob = mrz_date.strftime('%d/%m/%Y')
            except:
                ddmmyyyy_dob = raw_dob
        else:
            ddmmyyyy_dob = raw_dob
            
        _, ptc_code = calculate_passenger_type(ddmmyyyy_dob)
        
        # Format dates to DDMMMYY for output
        dob_formatted = _to_ddmmmyy(raw_dob)
        expiry_formatted = _to_ddmmmyy(raw_expiry)
        
        row = {
            "Last Name": item.get('surname', ''),
            "First Name and Middle Name": first_middle,
            "Title": title,
            "PTC": ptc_code, # Passenger Type Code based on age
            "Gender": sex,
            "Date of Birth": dob_formatted,
            "Passport Last Name": item.get('surname', ''),
            "Passport First Name": first_middle,
            "Passport Middle Name": "",
            "Passport Number": item.get('passport_number', ''),
            "Passport Nationality": item.get('nationality', '')[:3],
            "Passport Issue Country": item.get('issuing_country', '')[:3],
            "Passport Expiry Date": expiry_formatted,
            # Empty fields
            "Visa Number": "", "Visa Type": "", "Visa Issue Date": "", "Place of Birth": "",
            "Visa Place of Issue": "", "Visa Country of Application": "", "Address Type": "",
            "Address Country": "", "Address Details": "", "Address City": "",
            "Address State": "", "Address Zip Code": ""
        }
        formatted_rows.append(row)
        
    return pd.DataFrame(formatted_rows)

def export_to_spreadsheet(data_list, output_file, format='excel'):
    """
    Exports a list of dictionaries to a spreadsheet (Excel or CSV).
    """
    if not data_list:
        logger.warning("No data to export.")
        return False

    try:
        df = pd.DataFrame(data_list)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if format.lower() == 'excel' or output_file.endswith('.xlsx'):
            if not output_file.endswith('.xlsx'):
                output_file += '.xlsx'
            df.to_excel(output_file, index=False)
            logger.info(f"Data exported to {output_file}")
            
        elif format.lower() == 'csv' or output_file.endswith('.csv'):
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Data exported to {output_file}")
            
        else:
            logger.error(f"Unsupported format: {format}")
            return False
            
        return True

    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        return False
