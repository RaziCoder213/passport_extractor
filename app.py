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
from src.utils import parse_date


def process_pdf_file(uploaded_file, use_gpu=False, airline="flydubai"):
    """
    Process a single uploaded PDF file and extract passport data.

    Args:
        uploaded_file: Uploaded file object (Streamlit or similar)
        use_gpu (bool): Whether to use GPU acceleration
        airline (str): Airline format for date formatting ("flydubai", "default", "iraqi airways")

    Returns:
        List[dict]: Extracted passport data for each page/passport
    """
    extractor = PassportExtractor(use_gpu=use_gpu)
    results = []
    tmp_path = None

    try:
        # Save uploaded file to temporary location
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Extract passport data from PDF with airline-specific formatting
        results = extractor.process_pdf(tmp_path, airline=airline.lower(), progress_callback=None)
        if not results:
            print(f"⚠️ No passport data found in PDF: {uploaded_file.name}")

        # Attach source file name to each result
        for res in results:
            res['source_file'] = uploaded_file.name

    except Exception as e:
        print(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")
        results = []

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp file: {cleanup_error}")

    return results

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
    st.sidebar.info("💡 Running on Streamlit free tier - CPU only, max 10 MB per file")
    use_gpu = st.sidebar.checkbox("Enable GPU Acceleration", value=False, disabled=True)  # Disabled for free tier
    airline = st.sidebar.selectbox("Choose Airline Format", ["Default", "Iraqi Airways", "Flydubai"])

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Passport Files",
        type=['png', 'jpg', 'jpeg', 'pdf', 'avif', 'webp', 'bmp', 'tiff'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Extract Data"):
            # Initialize extractor inside the button click to use the latest settings
            extractor = PassportExtractor(use_gpu=use_gpu)
            
            all_results = []
            main_progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                st.write(f"Processing file {i+1} of {len(uploaded_files)}: {file.name}")
                
                # Check file size for Streamlit free tier
                file_size = len(file.getvalue())
                if file_size > 10 * 1024 * 1024:  # 10 MB
                    st.error(f"❌ File too large: {file.name} ({file_size / (1024*1024):.1f} MB). Max 10 MB allowed.")
                    main_progress_bar.progress((i + 1) / len(uploaded_files))
                    continue
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Check both MIME type and file extension for PDF detection
                    is_pdf = file.type == "application/pdf" or file.name.lower().endswith('.pdf')
                    
                    if is_pdf:
                        st.write(f"📄 Processing as PDF: {file.name} (type: {file.type})")
                        
                        try:
                            # Use the new standalone PDF processing function
                            results = process_pdf_file(file, use_gpu=use_gpu, airline=airline.lower())
                            if not results:
                                st.warning(f"⚠️ No passport data found in PDF: {file.name}")
                        except Exception as e:
                            st.error(f"❌ PDF processing failed for {file.name}: {str(e)}")
                            results = []
                            
                    else:
                        st.write(f"🖼️ Processing as image: {file.name} (type: {file.type})")
                        result = extractor.get_data(tmp_path, airline=airline.lower())
                        results = [result] if result else []
                        if not results:
                            st.warning(f"⚠️ No passport data found in image: {file.name}")
                    
                    for res in results:
                        res['source_file'] = file.name
                    all_results.extend(results)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
                    import traceback
                    st.error(f"Debug info: {traceback.format_exc()}")
                    
                finally:
                    # Safely remove temp file
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception as cleanup_error:
                        st.warning(f"Warning: Could not clean up temp file: {cleanup_error}")
                
                # Update main progress bar
                main_progress_bar.progress((i + 1) / len(uploaded_files))

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
                
                st.download_button(
                    label="Download data as Excel",
                    data=output.getvalue(),
                    file_name=f"passport_data_{airline.lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

if __name__ == "__main__":
    main()
