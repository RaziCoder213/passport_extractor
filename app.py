import streamlit as st
import os
import tempfile
import pandas as pd
import io
from PIL import Image
import pillow_heif

from src.extractor import PassportExtractor
from src.validator import validate_passport_data

pillow_heif.register_heif_opener()

st.set_page_config(page_title="Airline OCR Tool", layout="wide")

@st.cache_resource
def get_extractor(use_gpu, airline):
    return PassportExtractor(use_gpu=use_gpu, airline=airline)

def main():
    st.title("🛂 Airline Format Passport OCR")
    
    # Sidebar
    airline = st.sidebar.radio("Select Airline Format", ["iraqi", "flydubai"])
    use_gpu = st.sidebar.checkbox("Use GPU for OCR", value=False)
    
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

    if st.button("Process Files", type="primary") and uploaded_files:
        extractor = get_extractor(use_gpu, airline)
        processed_data = []

        for uploaded_file in uploaded_files:
            # Save and process
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            
            raw = extractor.get_data(path)
            if raw:
                # ✅ Validate (Unpacking 2 values)
                is_valid, errors = validate_passport_data(raw, airline)
                
                # ✅ Map to IRAQI Format
                if airline == "iraqi":
                    final_row = {
                        "TYPE": "Adult", # Manual logic can be added for Age
                        "TITLE": "MR" if raw["sex"] == "Male" else "MRS",
                        "FIRST NAME": raw["name"],
                        "LAST NAME": raw["surname"],
                        "DOB (DD/MM/YYYY)": raw["date_of_birth"],
                        "GENDER": raw["sex"],
                        "VALIDATION": "OK" if is_valid else "; ".join(errors)
                    }
                
                # ✅ Map to FLYDUBAI Format
                else:
                    final_row = {
                        "Last Name": raw["surname"],
                        "First Name and Middle Name": raw["name"],
                        "Title": "MR" if raw["sex"] == "Male" else "MRS",
                        "PTC": "ADT",
                        "Gender": "F" if raw["sex"] == "Female" else "M",
                        "Date of Birth": raw["date_of_birth"],
                        "Passport Number": raw["passport_number"],
                        "Passport Nationality": raw["nationality"],
                        "Passport Issue Country": raw["issuing_country"],
                        "Passport Expiry Date": raw["expiration_date"],
                        "VALIDATION": "OK" if is_valid else "; ".join(errors)
                    }
                processed_data.append(final_row)
            os.remove(path)

        if processed_data:
            df = pd.DataFrame(processed_data)
            st.subheader(f"Results: {airline.upper()} Format")
            st.dataframe(df)
            
            # Excel Download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("Download Excel", output.getvalue(), f"{airline}_manifest.xlsx")
        else:
            st.error("No data could be extracted.")

if __name__ == "__main__":
    main()
