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

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Passport Files",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Extract Data"):
            # Initialize extractor inside the button click to use the latest settings
            extractor = PassportExtractor(use_gpu=use_gpu)
            
            all_results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
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
                finally:
                    os.remove(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))

            if not all_results:
                st.warning("No data could be extracted. Please check the files or try again.")
                return

            # Format data based on airline selection
            if airline == "Iraqi Airways":
                df = format_iraqi_airways(all_results)
            elif airline == "Flydubai":
                df = format_flydubai(all_results)
            else:
                df = pd.DataFrame(all_results)

            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"passport_data_{airline.lower()}.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
