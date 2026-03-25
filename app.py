import os
import sys
import tempfile
from io import BytesIO

import gradio as gr
import pandas as pd

from src.extractor import PassportExtractor
from src.formats import format_fly_baghdad, format_fly_dubai, format_iraqi

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_FILE_TYPES = [".png", ".jpg", ".jpeg", ".pdf", ".avif", ".webp", ".bmp", ".tiff"]
EXTRACTOR = PassportExtractor(use_gpu=True)


def _safe_reason(problem):
    return (
        problem.get("reason")
        or problem.get("issue")
        or problem.get("message")
        or problem.get("error")
        or "<unspecified>"
    )


def _format_data(good_results, airline):
    if airline == "Fly Dubai":
        return format_fly_dubai(good_results)
    if airline == "Iraqi":
        return format_iraqi(good_results)
    if airline == "Fly Baghdad":
        return format_fly_baghdad(good_results)
    return pd.DataFrame(good_results)


def _placeholder_rows(airline, problematic_files, good_results):
    rows = []
    for problem in problematic_files:
        if airline == "Fly Dubai":
            rows.append(
                {
                    "TYPE": "Adult",
                    "TITLE": "MR",
                    "FIRST NAME": "•••",
                    "LAST NAME": "•••",
                    "DOB (DD/MM/YYYY)": "•••",
                    "GENDER": "Male",
                }
            )
        elif airline == "Iraqi":
            rows.append(
                {
                    "Last Name": "•••",
                    "First Name and Middle Name": "•••",
                    "Title": "MR",
                    "PTC": "ADT",
                    "Gender": "M",
                    "Date of Birth": "•••",
                    "Passport Last Name": "•••",
                    "Passport First Name": "•••",
                    "Passport Middle Name": "",
                    "Passport Number": "•••",
                    "Passport Nationality": "•••",
                    "Passport Issue Country": "•••",
                    "Passport Expiry Date": "•••",
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
                    "Address Zip Code": "",
                }
            )
        elif airline == "Fly Baghdad":
            rows.append(
                {
                    "Sequence": len(good_results) + len(rows) + 1,
                    "Traveling With": "",
                    "Pax Type": "ADT",
                    "Title": "MR",
                    "First Name": "•••",
                    "Last Name": "•••",
                    "Gender": "MALE",
                    "DOB (dd/mm/yyyy)": "•••",
                    "Nationality": "•••",
                    "Passport Number": "•••",
                    "Passport Expiry (dd/mm/yyyy)": "•••",
                    "Passport Issued Country": "•••",
                }
            )
        else:
            rows.append(
                {
                    "source_file": problem["file_name"],
                    "surname": "•••",
                    "given_names": "•••",
                    "passport_number": "•••",
                    "nationality": "•••",
                    "date_of_birth": "•••",
                    "sex": "•••",
                    "expiration_date": "•••",
                    "personal_number": "•••",
                    "mrz_found": False,
                }
            )
    return rows


def _write_export_files(df, airline):
    safe_airline = airline.lower().replace(" ", "_")
    out_dir = tempfile.mkdtemp(prefix="passport_exports_")
    csv_path = os.path.join(out_dir, f"passport_data_{safe_airline}.csv")
    xlsx_path = os.path.join(out_dir, f"passport_data_{safe_airline}.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    with pd.ExcelWriter(BytesIO(), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PassportData")
    df.to_excel(xlsx_path, index=False, sheet_name="PassportData")
    return csv_path, xlsx_path


def process_files(file_paths, airline, export_all_files, progress=gr.Progress(track_tqdm=False)):
    if not file_paths:
        return (
            "⚠️ Please upload at least one passport file.",
            "No failed file details.",
            None,
            None,
            None,
        )

    total_files = len(file_paths)
    all_results = []
    problematic_files = []
    progress(0, desc=f"Processing {total_files} files...")

    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()

        try:
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                problematic_files.append(
                    {
                        "file_name": file_name,
                        "reason": f"File too large ({file_size / (1024 * 1024):.1f} MB). Max 10 MB allowed.",
                    }
                )
                progress((i + 1) / total_files, desc=f"Skipping {file_name} (too large)")
                continue
        except Exception as e:
            problematic_files.append({"file_name": file_name, "reason": f"Error checking file size: {e}"})
            progress((i + 1) / total_files, desc=f"Skipping {file_name} (size error)")
            continue

        file_processed = False
        results = []

        try:
            if ext == ".pdf":
                results = EXTRACTOR.process_pdf(file_path, airline=airline.lower(), progress_callback=None) or []
                file_processed = True
            else:
                result = EXTRACTOR.get_data(file_path, airline=airline.lower())
                if result:
                    results = [result]
                    file_processed = True
        except Exception as e:
            problematic_files.append({"file_name": file_name, "reason": f"Processing error: {e}"})

        mrz_found_in_file = False
        for res in results:
            res["source_file"] = file_name
            if res.get("mrz_found", False):
                mrz_found_in_file = True
        all_results.extend(results)

        if not file_processed:
            problematic_files.append(
                {
                    "file_name": file_name,
                    "reason": "File could not be processed - unsupported format or corrupted file",
                }
            )
        elif not results:
            problematic_files.append(
                {
                    "file_name": file_name,
                    "reason": "No passport data detected - image may be blurry or passport not visible",
                }
            )
        elif not mrz_found_in_file:
            problematic_files.append(
                {
                    "file_name": file_name,
                    "reason": "MRZ data not found - passport may be damaged or partially visible",
                }
            )

        progress((i + 1) / total_files, desc=f"Processed {i + 1}/{total_files}: {file_name}")

    good_results = []
    for result in all_results:
        is_problematic = any(problem["file_name"] == result.get("source_file") for problem in problematic_files)
        if not is_problematic:
            good_results.append(result)

    successful_files = len(set(result.get("source_file", "") for result in good_results))
    failed_files = total_files - successful_files

    if failed_files > 0:
        status_md = f"⚠️ {successful_files}/{total_files} passports were successfully extracted."
    else:
        status_md = f"✅ All {successful_files} passports were successfully imported."

    if problematic_files:
        lines = ["**Failed Passports:**"]
        for problem in problematic_files:
            display_name = None
            for res in all_results:
                if res.get("source_file") == problem.get("file_name"):
                    display_name = (
                        res.get("surname")
                        or res.get("given_names")
                        or res.get("name")
                        or res.get("Last Name")
                        or res.get("First Name and Middle Name")
                    )
                    break
            if display_name:
                lines.append(f"- {problem.get('file_name', '<unknown>')} - {display_name} - {_safe_reason(problem)}")
            else:
                lines.append(f"- {problem.get('file_name', '<unknown>')} - {_safe_reason(problem)}")
        failed_md = "\n".join(lines)
    else:
        failed_md = "No failed files."

    if not good_results:
        return (
            f"{status_md}\n\n⚠️ No data could be extracted. Please check the files or try again.",
            failed_md,
            None,
            None,
            None,
        )

    df = _format_data(good_results, airline)
    if export_all_files and problematic_files:
        placeholders = _placeholder_rows(airline, problematic_files, good_results)
        if placeholders:
            df = pd.concat([df, pd.DataFrame(placeholders)], ignore_index=True)

    csv_path, xlsx_path = _write_export_files(df, airline)
    return status_md, failed_md, df, csv_path, xlsx_path


def clear_results():
    return "", "No failed file details.", None, None, None


def build_app():
    with gr.Blocks(title="Passport OCR Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛂 Passport OCR Extractor")
        gr.Markdown(
            "This tool extracts data from passport MRZ (Machine-Readable Zone) codes.\n"
            "Upload passport images or PDFs, and the app returns a structured table.\n"
            "You can also select an airline-specific output format."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                gr.Markdown("💡 Optimized for performance - GPU enabled")
                airline = gr.Dropdown(
                    choices=["Default", "Fly Dubai", "Iraqi", "Fly Baghdad"],
                    value="Default",
                    label="Choose Airline Format",
                )
                export_all_files = gr.Checkbox(
                    value=False,
                    label="Include all files in export",
                    info="When checked, includes files with missing/invalid data (shows ••• for missing fields)",
                )
            with gr.Column(scale=3):
                files = gr.File(
                    label="Upload Passport Files",
                    file_count="multiple",
                    file_types=ALLOWED_FILE_TYPES,
                    type="filepath",
                )
                with gr.Row():
                    extract_btn = gr.Button("Extract Data", variant="primary")
                    clear_btn = gr.Button("Clear Results")

                status = gr.Markdown("")
                with gr.Accordion("📋 Click to see failed passport details", open=False):
                    failed_details = gr.Markdown("No failed file details.")
                results_df = gr.Dataframe(label="📊 Extracted Data", interactive=False)

                with gr.Row():
                    csv_file = gr.File(label="Download data as CSV", interactive=False)
                    excel_file = gr.File(label="Download data as Excel", interactive=False)

        extract_btn.click(
            fn=process_files,
            inputs=[files, airline, export_all_files],
            outputs=[status, failed_details, results_df, csv_file, excel_file],
            queue=True,
        )
        clear_btn.click(
            fn=clear_results,
            inputs=[],
            outputs=[status, failed_details, results_df, csv_file, excel_file],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
    )
