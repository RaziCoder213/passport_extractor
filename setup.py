from setuptools import setup, find_packages

setup(
    name="passport-ocr-tool",
    version="1.0.0",
    description="A tool to extract data from passports (Images/PDFs) and export to Excel/CSV.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'passporteye',
        'easyocr',
        'matplotlib',
        'opencv-python',
        'python-dateutil',
        'pdf2image',
        'pandas',
        'openpyxl',
        'tqdm',
        'Pillow',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'passport-ocr=main:main',
        ],
    },
    python_requires='>=3.7',
)

