import pdfplumber
import os

PDF_DIR = "../syllabus_pdfs"
TEXT_DIR = "../syllabus_text"

os.makedirs(TEXT_DIR, exist_ok=True)

def extract_text(pdf_file, output_txt):
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[OK] Extracted text from {pdf_file}")

for pdf in os.listdir(PDF_DIR):
    if pdf.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, pdf)
        txt_name = pdf.replace(".pdf", ".txt")
        txt_path = os.path.join(TEXT_DIR, txt_name)
        extract_text(pdf_path, txt_path)
