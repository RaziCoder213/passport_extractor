import streamlit as st
import os
import tempfile
import pandas as pd
import io

from PIL import Image
import pillow_heif

from src.extractor import PassportExtractor
from src.validators import validate_passport_data

# --------------------------------------------------
# Enable HEIC / HEIF support
# --------------------------------------------------
pillow_heif.register_heif_opener()

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Passport OCR Tool",
    page_icon="🛂",
    layout="wide"
)

# --------------------------------------------------
# Cached Extractor
# --------------------------------------------------
@st.cache_resource
def get_extractor(use_gpu: bool, airline: str):
    return PassportExtractor(use_gpu=use_gpu, airline=airline)

# --------------------------------------------------
# Helper: Save & normalize uploaded files
# --------------------------------------------------
def save_uploaded_file(uploaded_file):
    """
    Saves uploaded file.
    Converts AVIF / HEIC / HEIF / WEBP → PNG for OCR compatibility.
    Returns path to temp file.
    """
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Convert unsupported image formats to PNG
        if ext in [".avif", ".heic", ".heif", ".webp", ".tiff", ".bmp"]:
            image = Image.open(tmp_path).convert("RGB")

            png_tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".png"
            )
            image.save(png_tmp.name, format="PNG")

            os.remove(tmp_path)
            return png_tmp.name

        return tmp_path

    except Exception as e:
        st.error(f"File handling error: {e}")
        return None

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():
    st.title("🛂 Passport OCR Extractor")

    st.markdown(
        """
        Upload passport files in **PDF or any image format**.
        The system automatically normalizes images and extracts MRZ data.
        """
    )

    # --------------------------------------------------
    # Sidebar
    # --------------------------------------------------
    st.sidebar.header("Configuration")

    airline = st.sidebar.radio(
        "Select Airline Format",
        ["iraqi", "flydubai"]
    )

    enable_validation = st.sidebar.checkbox(
        "Enable Data Validation", value=True
    )

    use_gpu = st.sidebar.checkbox(
        "Use GPU for OCR", value=False
    )

    # --------------------------------------------------
    # File Upload (ALL formats)
    # --------------------------------------------------
    uploaded_files = st.file_uploader(
        "Upload Passport Files",
        type=[
            "pdf",
            "jpg", "jpeg", "png",
            "avif", "heic", "heif",
            "webp", "tiff", "bmp"
        ],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload passport files to continue.")
        return

    st.success(f"{len(uploaded_files)} file(s) uploaded")

    # --------------------------------------------------
    # Process
    # --------------------------------------------------
    if st.button("Process Files", type="primary"):

        extractor = get_extractor(use_gpu, airline)

        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status.text(f"Processing: {uploaded_file.name}")

            temp_path = save_uploaded_file(uploaded_file)
            if not temp_path:
                continue

            try:
                ext = os.path.splitext(temp_path)[1].lower()
                file_results = []

                if ext == ".pdf":
                    file_results = extractor.process_pdf(temp_path)
                else:
                    data = extractor.get_data(temp_path)
                    if data:
                        file_results = [data]

                if enable_validation:
                    for res in file_results:
                        errors = validate_passport_data(
                            res, airline=airline
                        )
                        res["validation_errors"] = (
                            "; ".join(errors) if errors else "Valid"
                        )

                for res in file_results:
                    res["original_filename"] = uploaded_file.name
                    res["airline_format"] = airline

                results.extend(file_results)

            except Exception as e:
                st.error(f"Failed: {uploaded_file.name} — {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            progress.progress((idx + 1) / len(uploaded_files))

        status.text("Processing complete")

        # --------------------------------------------------
        # Results
        # --------------------------------------------------
        if not results:
            st.warning("No data extracted.")
            return

        df = pd.DataFrame(results)

        preferred_cols = [
            "surname",
            "name",
            "passport_number",
            "nationality",
            "date_of_birth",
            "sex",
            "expiration_date",
            "validation_errors",
            "airline_format",
            "original_filename",
        ]

        ordered_cols = preferred_cols + [
            c for c in df.columns if c not in preferred_cols
        ]

        df = df[[c for c in ordered_cols if c in df.columns]]

        st.dataframe(df, use_container_width=True)

        # --------------------------------------------------
        # Downloads
        # --------------------------------------------------
        col1, col2 = st.columns(2)

        csv_data = df.to_csv(index=False).encode("utf-8")
        col1.download_button(
            "⬇ Download CSV",
            data=csv_data,
            file_name=f"passport_data_{airline}.csv",
            mime="text/csv"
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)

        col2.download_button(
            "⬇ Download Excel",
            data=buffer.getvalue(),
            file_name=f"passport_data_{airline}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    main()
