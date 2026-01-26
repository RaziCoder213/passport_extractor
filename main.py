import streamlit as st
import os
import tempfile
from tqdm import tqdm

from src.extractor import PassportExtractor
from src.formats import export_to_spreadsheet
from src.validators import validate_passport_data
from src.utils import setup_logger
from config.settings import ALLOWED_EXTENSIONS

logger = setup_logger()

st.set_page_config(page_title="Passport OCR Tool", layout="wide")

st.title("Passport OCR Processing System")

# ----------------------------
# Sidebar Options (same as CLI args)
# ----------------------------
with st.sidebar:
    st.header("Processing Options")

    airline = st.radio(
        "Select Airline Format",
        options=["iraqi", "flydubai"],
        index=0
    )

    output_format = st.selectbox(
        "Output Format",
        options=["excel", "csv"]
    )

    use_gpu = st.checkbox("Use GPU for OCR", value=False)

# ----------------------------
# File Upload
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload Passport Files (PDF / JPG / PNG)",
    type=[ext.replace(".", "") for ext in ALLOWED_EXTENSIONS],
    accept_multiple_files=True
)

# ----------------------------
# Processing Trigger
# ----------------------------
if st.button("Process Passports"):

    if not uploaded_files:
        st.warning("Please upload at least one passport file.")
        st.stop()

    st.info(f"Processing {len(uploaded_files)} file(s) using {airline.upper()} format")

    extractor = PassportExtractor(
        use_gpu=use_gpu,
        airline=airline
    )

    extracted_results = []

    with st.status("Processing files...", expanded=True) as status:
        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(uploaded_file.name)[1].lower()

            if suffix not in ALLOWED_EXTENSIONS:
                st.warning(f"Unsupported file: {uploaded_file.name}")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                if suffix == ".pdf":
                    results = extractor.process_pdf(tmp_path)
                    extracted_results.extend(results)
                else:
                    result = extractor.get_data(tmp_path)
                    if result:
                        extracted_results.append(result)

            except Exception as e:
                logger.error(e)
                st.error(f"Failed to process {uploaded_file.name}")

            finally:
                os.unlink(tmp_path)

        status.update(label="Processing complete", state="complete")

    # ----------------------------
    # Validation
    # ----------------------------
    valid_count = 0

    for res in extracted_results:
        errors = validate_passport_data(
            res,
            airline=airline
        )

        if not errors:
            valid_count += 1
        else:
            res["validation_errors"] = "; ".join(errors)

    st.success(
        f"Extracted {len(extracted_results)} record(s) — {valid_count} valid"
    )

    # ----------------------------
    # Export
    # ----------------------------
    if extracted_results:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "passport_data")

            export_to_spreadsheet(
                extracted_results,
                output_path,
                format=output_format,
                airline=airline
            )

            file_path = f"{output_path}.{output_format}"

            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download Result",
                    data=f,
                    file_name=f"passport_data_{airline}.{output_format}",
                    mime="application/octet-stream"
                )
    else:
        st.warning("No valid data extracted.")
