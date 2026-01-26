import streamlit as st
import os
import tempfile
import pandas as pd
import io
from src.extractor import PassportExtractor

st.set_page_config(page_title="Passport OCR", layout="wide")

def main():
    st.title("🛂 Passport OCR Manifest Tool")
    
    airline = st.sidebar.radio("Format", ["iraqi", "flydubai"])
    use_gpu = st.sidebar.checkbox("Use GPU")
    uploaded_files = st.file_uploader("Upload Passports", accept_multiple_files=True)

    if st.button("Process") and uploaded_files:
        extractor = PassportExtractor(use_gpu=use_gpu, airline=airline)
        results = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            
            # The extractor now returns the dictionary with the CORRECT keys already!
            data = extractor.get_data(path)
            if data:
                results.append(data)
            
            os.remove(path)

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("Download Manifest", output.getvalue(), f"{airline}_manifest.xlsx")

if __name__ == "__main__":
    main()
