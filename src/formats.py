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

def format_flydubai(data_list):
    """
    Formats data for Flydubai template.
    Columns: Last Name, First Name and Middle Name, Title, PTC, Gender, Date of Birth, 
             Passport Last Name, Passport First Name, Passport Middle Name, Passport Number, 
             Passport Nationality, Passport Issue Country, Passport Expiry Date, 
             Visa Number, Visa Type, Visa Issue Date, Place of Birth, Visa Place of Issue, 
             Visa Country of Application, Address Type, Address Country, Address Details, 
             Address City, Address State, Address Zip Code
    """
    formatted_rows = []
    
    for item in data_list:
        # Determine Title and PTC (Passenger Type Code) based on Age/Gender
        # Defaulting to ADT (Adult) if dob is missing or calculation complex for now.
        # Logic: 
        # Infant < 2 years
        # Child < 12 years
        # Adult >= 12 years
        # Title: MR, MRS, MISS, MSTR
        
        # We need to parse DOB properly to determine age.
        # For now, we will map available fields.
        
        sex = item.get('sex', '').upper()
        # Simple Title Logic
        title = "MR" if sex == 'M' else "MRS" 
        
        # Split names
        # 'name' field usually contains First + Middle names in MRZ
        first_middle = item.get('name', '')
        
        row = {
            "Last Name": item.get('surname', ''),
            "First Name and Middle Name": first_middle,
            "Title": title,
            "PTC": "ADT", # Default to Adult
            "Gender": sex,
            "Date of Birth": item.get('date_of_birth', ''),
            "Passport Last Name": item.get('surname', ''),
            "Passport First Name": first_middle, # Using full given name
            "Passport Middle Name": "", # MRZ doesn't separate Middle name clearly
            "Passport Number": item.get('passport_number', ''),
            "Passport Nationality": item.get('nationality', '')[:3], # Usually 3 letter code
            "Passport Issue Country": item.get('issuing_country', '')[:3],
            "Passport Expiry Date": item.get('expiration_date', ''),
            # Empty fields for Visa/Address as they are not in Passport MRZ
            "Visa Number": "",
            "Visa Type": "",
            "Visa Issue Date": "",
            "Place of Birth": "",
            "Visa Place of Issue": "",
            "Visa Country of Application": "",
            "Address Type": "",
            "Address Country": "",
            "Address Details": "",
            "Address City": "",
            "Address State": "",
            "Address Zip Code": ""
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
