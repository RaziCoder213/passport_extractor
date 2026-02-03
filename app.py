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
def get_extractor():
    return PassportExtractor(use_gpu=False)

extractor = get_extractor()

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
    st.markdown("""
    Upload passport images (JPG, PNG) or PDFs to extract MRZ data automatically.
    The tool extracts details like Name, Surname, Passport Number, and Date of Birth.
    """)

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    enable_validation = st.sidebar.checkbox("Enable Data Validation", value=True)
    
    # Format Selection
    st.sidebar.subheader("Export Format")
    export_format_type = st.sidebar.radio(
        "Choose Airline Format",
        ("Standard", "Iraqi Airways", "Flydubai"),
        index=0
    )
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Choose passport files", 
        type=['png', 'jpg', 'jpeg', 'pdf'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Loaded {len(uploaded_files)} files. Click 'Process Files' to start.")
        
        if st.button("Process Files"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save to temp file because extractor needs a path
                temp_path = save_uploaded_file(uploaded_file)
                
                if temp_path:
                    try:
                        # Determine if PDF based on MIME type or extension
                        is_pdf = False
                        if uploaded_file.type == "application/pdf":
                            is_pdf = True
                        elif os.path.splitext(temp_path)[1].lower() == '.pdf':
                            is_pdf = True
                        
                        file_results = []
                        
                        if is_pdf:
                            file_results = extractor.process_pdf(temp_path)
                        else:
                            # Assume it's an image
                            data = extractor.get_data(temp_path)
                            if data:
                                file_results = [data]
                        
                        # Add validation if enabled
                        if enable_validation:
                            for res in file_results:
                                errors = validate_passport_data(res)
                                res['validation_errors'] = "; ".join(errors) if errors else "Valid"
                        
                        # Add original filename for reference (since we used a temp file)
                        for res in file_results:
                            res['original_filename'] = uploaded_file.name
                            
                        results.extend(file_results)
                        
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            if results:
                st.success(f"Successfully extracted {len(results)} records.")
                
                # Apply formatting based on selection
                if export_format_type == "Iraqi Airways":
                    df = format_iraqi_airways(results)
                    st.info("Applying Iraqi Airways format")
                elif export_format_type == "Flydubai":
                    df = format_flydubai(results)
                    st.info("Applying Flydubai format")
                else:
                    # Standard Format
                    df = pd.DataFrame(results)
                    # Reorder columns for better readability if possible
                    cols = ['surname', 'name', 'passport_number', 'nationality', 'date_of_birth', 'sex', 'expiration_date', 'validation_errors', 'original_filename']
                    # Add missing cols to list if they exist in df
                    all_cols = cols + [c for c in df.columns if c not in cols]
                    # Filter valid columns only
                    final_cols = [c for c in all_cols if c in df.columns]
                    df = df[final_cols]
                
                # Display Data
                st.dataframe(df)
                
                # Download Buttons
                col1, col2 = st.columns(2)
                
                # CSV Download
                csv = df.to_csv(index=False).encode('utf-8')
                col1.download_button(
                    label=f"Download CSV ({export_format_type})",
                    data=csv,
                    file_name=f"passport_data_{export_format_type.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )
                
                # Excel Download (requires openpyxl)
                # Streamlit might need a BytesIO buffer for Excel
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                
                col2.download_button(
                    label=f"Download Excel ({export_format_type})",
                    data=buffer.getvalue(),
                    file_name=f"passport_data_{export_format_type.lower().replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                
            else:
                st.warning("No data extracted from the provided files.")

if __name__ == "__main__":
    main()
