# Passport OCR Tool

A robust tool to extract data from passport images (PNG, JPG) and PDFs using OCR. It extracts MRZ (Machine Readable Zone) data and exports it to Excel or CSV formats.

## Features

- **Multi-format Support**: Handles images (JPG, PNG) and PDFs.
- **Batch Processing**: Process single files or entire directories.
- **Data Export**: Save results to Excel (`.xlsx`) or CSV.
- **Validation**: Basic validation of extracted fields.
- **Web Interface**: User-friendly web app for easy demonstration.
- **Local Processing**: Runs entirely on your local machine.

## Prerequisites

### 1. Python
Ensure you have Python 3.7+ installed.

### 2. System Dependencies (Poppler)
This tool requires `poppler` for PDF processing.

- **macOS** (using Homebrew):
  ```bash
  brew install poppler
  ```

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install poppler-utils
  ```

- **Windows**:
  1. Download the latest binary from [@oschwartz10612 version](https://github.com/oschwartz10612/poppler-windows/releases/).
  2. Extract the zip file.
  3. Add the `bin/` folder to your System PATH environment variable.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd passport-ocr-tool
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Web App (Recommended for Demos)
Launch the easy-to-use web interface:

```bash
python app.py
```
This will open the tool in your default web browser (usually at `http://127.0.0.1:7860`). You can drag and drop files and download the results as Excel/CSV.

### 2. Command Line Interface (CLI)

Run the tool using `main.py`:

```bash
# Process a single image
python main.py --input path/to/passport.jpg

# Process a directory of images/PDFs
python main.py --input path/to/folder --output my_results

# Export to CSV
python main.py --input path/to/folder --format csv
```

### Command Line Arguments

- `--input`, `-i`: Path to input file or directory (Required).
- `--output`, `-o`: Output filename (default: `passport_data`).
- `--format`, `-f`: Output format: `excel` or `csv` (default: `excel`).
- `--gpu`: Enable GPU acceleration for OCR (requires CUDA).

## Project Structure

```
passport-ocr-tool/
├── src/
│   ├── extractor.py      # Core extraction logic
│   ├── utils.py          # Helper functions
│   ├── validators.py     # Data validation
│   └── formats.py        # Export handlers
├── config/
│   └── settings.py       # Configuration
├── app.py                # Gradio Web App
├── main.py               # CLI Entry point
└── ...
```

## Troubleshooting

- **PDF Error**: "Unable to get page count" or "poppler is not installed".
  - Ensure `poppler` is installed and in your PATH.
- **OCR Accuracy**:
  - Ensure images are clear and high resolution.
  - The MRZ zone (bottom lines) must be visible and unobstructed.
