import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

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
        
        row = {
            "TYPE": "Adult", # Default
            "TITLE": title,
            "FIRST NAME": item.get('name', ''),
            "LAST NAME": item.get('surname', ''),
            "DOB (DD/MM/YYYY)": item.get('date_of_birth', ''),
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
        # Format to DDMMMYY
        return dt_obj.strftime('%d%b%y').upper()
    except ValueError:
        return date_str # Return original if parsing fails

def format_flydubai(data_list):
    """
    Formats data for Flydubai template.
    - Date of Birth: DDMMMYY
    - Passport Expiry Date: DOB repeated four times
    """
    formatted_rows = []
    
    for item in data_list:
        sex = item.get('sex', '').upper()
        title = "MR" if sex == 'M' else "MRS" 
        
        first_middle = item.get('name', '')
        
        # Format dates
        dob_formatted = _to_ddmmmyy(item.get('date_of_birth', ''))
        expiry_formatted = dob_formatted * 4
        
        row = {
            "Last Name": item.get('surname', ''),
            "First Name and Middle Name": first_middle,
            "Title": title,
            "PTC": "ADT", # Default to Adult
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
