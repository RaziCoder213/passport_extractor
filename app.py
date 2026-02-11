import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import tempfile
import pandas as pd
import time
from src.extractor import PassportExtractor
from src.validators import validate_passport_data
from src.formats import format_iraqi_airways, format_flydubai

# Set page configuration
st.set_page_config(
    page_title="Passport OCR Tool",
    page_icon="🛂",
    layout="wide"
)

# Initialize Extractor (cached to avoid reloading model)
@st.cache_resource
def get_extractor(use_gpu=False):
    return PassportExtractor(use_gpu=use_gpu)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path."""
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    st.title("🛂 Passport OCR Extractor")
    
    # Add a description
    st.markdown("""
    This tool extracts data from passport MRZ (Machine-Readable Zone) codes. 
    Upload passport images or PDFs, and the app will return a structured table of the extracted information. 
    You can also select an airline-specific format for the output data.
    """)

    # Sidebar Configuration
    st.sidebar.header("Settings")
    use_gpu = st.sidebar.checkbox("Enable GPU Acceleration", value=False)
    airline = st.sidebar.selectbox("Choose Airline Format", ["Default", "Iraqi Airways", "Flydubai"])

    # Initialize session state for results
    if 'raw_results' not in st.session_state:
        st.session_state.raw_results = None

    # File Uploader with enhanced configuration
    st.info("💡 You can upload up to 500 passport files at once. Supported formats: PNG, JPG, JPEG, PDF, AVIF, WEBP, BMP, TIFF")
    
    uploaded_files = st.file_uploader(
        "Upload Passport Files (Max 500 files)",
        type=['png', 'jpg', 'jpeg', 'pdf', 'avif', 'webp', 'bmp', 'tiff'],
        accept_multiple_files=True,
        key="passport_uploader"
    )

    if uploaded_files:
        file_count = len(uploaded_files)
        if file_count > 500:
            st.error(f"❌ You have uploaded {file_count} files. Maximum allowed is 500 files. Please select fewer files.")
            st.stop()
        
        if file_count > 0:
            st.success(f"📁 {file_count} file(s) uploaded successfully!")
            
            if st.button("Extract Data", type="primary"):
                # Initialize extractor inside the button click to use the latest settings
                extractor = PassportExtractor(use_gpu=use_gpu)
                
                all_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process files in batches for better performance
                batch_size = 10
                
                for i, file in enumerate(uploaded_files):
                    # Update status
                    status_text.text(f"Processing file {i+1} of {file_count}: {file.name}")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        if file.type == "application/pdf":
                            results = extractor.process_pdf(tmp_path)
                        else:
                            result = extractor.get_data(tmp_path)
                            results = [result] if result else []
                        
                        for res in results:
                            res['source_file'] = file.name
                        all_results.extend(results)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error processing {file.name}: {str(e)}")
                        
                    finally:
                        os.remove(tmp_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / file_count)
                
                # Clear status text
                status_text.empty()

            if not all_results:
                st.warning("No data could be extracted. Please check the files or try again.")
                st.session_state.raw_results = None
            else:
                st.session_state.raw_results = all_results
                st.success("Extraction Complete!")

    # Display and Download (if results exist)
    if st.session_state.raw_results:
        # Format data based on airline selection
        if airline == "Iraqi Airways":
            df = format_iraqi_airways(st.session_state.raw_results)
        elif airline == "Flydubai":
            df = format_flydubai(st.session_state.raw_results)
        else:
            df = pd.DataFrame(st.session_state.raw_results)

        st.dataframe(df)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"passport_data_{airline.lower()}.csv",
                mime="text/csv",
            )
        
        with col2:
            # Create an in-memory Excel file
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='PassportData')
                
                # Access the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['PassportData']
                
                # Iterate over all columns to adjust width and format
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter # Get the column name
                    header_value = column[0].value
                    
                    # Check if this is a date column for Flydubai or generic
                    is_date_col = header_value in ["Date of Birth", "Passport Expiry Date", "DOB (DD/MM/YYYY)"]
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                            
                            # If it's a date column, force text format to prevent Excel from auto-converting to default date format
                            if is_date_col and cell.row > 1: # Skip header
                                cell.number_format = '@'
                        except:
                            pass
                    
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            st.download_button(
                label="Download data as Excel",
                data=output.getvalue(),
                file_name=f"passport_data_{airline.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

if __name__ == "__main__":
    main()
