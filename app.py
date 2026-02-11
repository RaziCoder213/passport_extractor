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


def process_pdf_file(uploaded_file, airline="flydubai"):
    """
    Process a single uploaded PDF file and extract passport data.

    Args:
        uploaded_file: Uploaded file object (Streamlit or similar)
        airline (str): Airline format for date formatting ("flydubai", "default", "iraqi airways")

    Returns:
        List[dict]: Extracted passport data for each page/passport
    """
    extractor = PassportExtractor(use_gpu=True)
    results = []
    tmp_path = None

    try:
        # Save uploaded file to temporary location
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Extract passport data from PDF with airline-specific formatting
        try:
            results = extractor.process_pdf(tmp_path, airline=airline.lower(), progress_callback=None)
            if not results:
                print(f"⚠️ No passport data found in PDF: {uploaded_file.name}")
                # Return placeholder result for PDFs without MRZ
                results = [{
                    "surname": "•••",
                    "name": "•••",
                    "country": "•••",
                    "nationality": "•••",
                    "passport_number": "•••",
                    "sex": "•••",
                    "date_of_birth": "•••",
                    "expiration_date": "•••",
                    "personal_number": "•••",
                    "mrz_full_string": "",
                    "valid_score": 0,
                    "mrz_found": False
                }]
        except Exception as e:
            print(f"⚠️ Error processing PDF {uploaded_file.name}: {str(e)}")
            # Return placeholder result for PDFs with processing errors
            results = [{
                "surname": "•••",
                "name": "•••",
                "country": "•••",
                "nationality": "•••",
                "passport_number": "•••",
                "sex": "•••",
                "date_of_birth": "•••",
                "expiration_date": "•••",
                "personal_number": "•••",
                "mrz_full_string": "",
                "valid_score": 0,
                "mrz_found": False
            }]

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
def get_extractor():
    return PassportExtractor(use_gpu=True)

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
    st.sidebar.info("💡 Optimized for performance - GPU enabled")
    airline = st.sidebar.selectbox("Choose Airline Format", ["Default", "Iraqi Airways", "Flydubai"])

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Passport Files",
        type=['png', 'jpg', 'jpeg', 'pdf', 'avif', 'webp', 'bmp', 'tiff'],
        accept_multiple_files=True
    )

    # Initialize session state to prevent crashes from rapid clicks
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    if uploaded_files and not st.session_state.processing:
        if st.button("Extract Data"):
            st.session_state.processing = True
            # Reset export decisions for new processing
            st.session_state.export_decision_made = False
            st.session_state.include_problematic_files = False
            try:
                # Initialize extractor inside the button click to use the latest settings
                extractor = PassportExtractor(use_gpu=True)
                
                all_results = []
                problematic_files = []  # Track files with issues
                
                # Create a container for processing status
                status_container = st.container()
                progress_container = st.container()
                
                with progress_container:
                    main_progress_bar = st.progress(0)
                
                # Show single status line at the beginning
                with status_container:
                    status_text = st.empty()
                    status_text.write(f"🔄 Processing {len(uploaded_files)} files...")
                
                for i, file in enumerate(uploaded_files):
                    
                    # Check file size for Streamlit free tier
                    try:
                        file_size = len(file.getvalue())
                        if file_size > 10 * 1024 * 1024:  # 10 MB
                            st.error(f"❌ File too large: {file.name} ({file_size / (1024*1024):.1f} MB). Max 10 MB allowed.")
                            with progress_container:
                                main_progress_bar.progress((i + 1) / len(uploaded_files))
                            continue
                    except Exception as e:
                        st.error(f"❌ Error checking file size for {file.name}: {str(e)}")
                        with progress_container:
                            main_progress_bar.progress((i + 1) / len(uploaded_files))
                        continue
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        # Check both MIME type and file extension for PDF detection
                        is_pdf = file.type == "application/pdf" or file.name.lower().endswith('.pdf')
                        
                        file_processed = False
                        if is_pdf:
                            try:
                                # Use the new standalone PDF processing function
                                results = process_pdf_file(file, airline=airline.lower())
                                if results:
                                    file_processed = True
                            except Exception as e:
                                results = []
                                pass
                        else:
                            try:
                                result = extractor.get_data(tmp_path, airline=airline.lower())
                                if result:
                                    results = [result]
                                    file_processed = True
                                else:
                                    results = []
                            except Exception as e:
                                results = []
                                pass
                        
                        # Add source file to results and extend all_results
                        for res in results:
                            res['source_file'] = file.name
                            # Add status field to indicate if MRZ was found
                            if 'mrz_found' not in res:
                                res['mrz_found'] = True
                        all_results.extend(results)
                        
                        # Track problematic files
                        if not file_processed or not results:
                            problematic_files.append({
                                'file_name': file.name,
                                'issue': 'No MRZ data found - image may be blurry or passport not detected'
                            })
                            # Debug output
                            print(f"DEBUG: Added problematic file: {file.name} (file_processed: {file_processed}, results: {len(results)})")
                        
                    except Exception as e:
                        # Track files that caused errors
                        problematic_files.append({
                            'file_name': file.name,
                            'issue': f'Processing error: {str(e)}'
                        })
                        pass
                        
                    finally:
                        # Safely remove temp file
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except:
                            pass
                    
                    # Update main progress bar - use container to avoid state issues
                    with progress_container:
                        main_progress_bar.progress((i + 1) / len(uploaded_files))

                # Update status to show completion
                with status_container:
                    status_text.write(f"✅ Processing complete!")
                
                # Debug output for problematic files
                print(f"DEBUG: Total problematic files: {len(problematic_files)}")
                print(f"DEBUG: Total results: {len(all_results)}")
                for problem in problematic_files:
                    print(f"DEBUG: Problematic: {problem['file_name']} - {problem['issue']}")

                # Show results only after all processing is complete
                successful_files = len(set(result.get('source_file', '') for result in all_results))
                st.success(f"📋 Extracted data of {successful_files} files")
                
                # Show warning if there are problematic files
                if problematic_files:
                    st.warning(f"⚠️ {len(problematic_files)} file(s) could not be processed properly. Click export for details.")

                if not all_results:
                    st.warning("No data could be extracted. Please check the files or try again.")
                    st.session_state.processing = False
                    return

                # Format data based on airline selection
                if airline == "Iraqi Airways":
                    df = format_iraqi_airways(all_results)
                elif airline == "Flydubai":
                    df = format_flydubai(all_results)
                else:
                    df = pd.DataFrame(all_results)

                st.dataframe(df)
                
                # Handle problematic files export popup
                print(f"DEBUG: About to check problematic_files popup. Length: {len(problematic_files)}")
                if problematic_files:
                    print(f"DEBUG: Showing popup for {len(problematic_files)} problematic files")
                    # Initialize session state for export decision if not exists
                    if 'export_decision_made' not in st.session_state:
                        st.session_state.export_decision_made = False
                    if 'include_problematic_files' not in st.session_state:
                        st.session_state.include_problematic_files = False
                    
                    # Show export confirmation dialog
                    if not st.session_state.export_decision_made:
                        st.subheader("⚠️ Export Confirmation Required")
                        st.write(f"**{len(problematic_files)} file(s) could not be processed properly:**")
                        
                        # Show problematic files details
                        with st.expander(f"Click to see details of {len(problematic_files)} problematic file(s)"):
                            for problem in problematic_files:
                                st.write(f"• **{problem['file_name']}**: {problem['issue']}")
                        
                        st.write("**Would you like to include these files in your export?**")
                        st.write("• **Yes**: Files will show dots (•••) for missing data")
                        st.write("• **No**: Only successfully processed files will be exported")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Yes, Include All Files", type="primary", use_container_width=True):
                                st.session_state.include_problematic_files = True
                                st.session_state.export_decision_made = True
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ No, Export Only Good Files", type="secondary", use_container_width=True):
                                st.session_state.include_problematic_files = False
                                st.session_state.export_decision_made = True
                                st.rerun()
                    
                    # Create export dataframe based on user's decision
                    if st.session_state.export_decision_made:
                        if st.session_state.include_problematic_files:
                            # Create dataframe with placeholder data for problematic files
                            placeholder_data = []
                            for problem in problematic_files:
                                placeholder_data.append({
                                    'source_file': problem['file_name'],
                                    'surname': '•••',
                                    'given_names': '•••',
                                    'passport_number': '•••',
                                    'nationality': '•••',
                                    'date_of_birth': '•••',
                                    'sex': '•••',
                                    'expiration_date': '•••',
                                    'personal_number': '•••',
                                    'mrz_found': False
                                })
                            
                            # Combine successful and placeholder data
                            if placeholder_data:
                                placeholder_df = pd.DataFrame(placeholder_data)
                                export_df = pd.concat([df, placeholder_df], ignore_index=True)
                            else:
                                export_df = df
                        else:
                            export_df = df
                        
                        # Add reset button to change decision
                        if st.button("🔄 Change Export Decision"):
                            st.session_state.export_decision_made = False
                            st.rerun()
                else:
                    export_df = df
                
                # Download buttons - moved inside try block where df is guaranteed to be defined
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = export_df.to_csv(index=False).encode('utf-8')
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
                        export_df.to_excel(writer, index=False, sheet_name='PassportData')
                    
                    st.download_button(
                        label="Download data as Excel",
                        data=output.getvalue(),
                        file_name=f"passport_data_{airline.lower()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                
                # Reset processing state
                st.session_state.processing = False
                
            except Exception as e:
                st.error(f"❌ Unexpected error during processing: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")
                st.session_state.processing = False
                
            finally:
                # Ensure processing state is reset even if an error occurs
                if 'processing' in st.session_state:
                    st.session_state.processing = False

if __name__ == "__main__":
    main()
