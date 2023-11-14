import os
from PyPDF2 import PdfReader

def pdf_text(folder):
    company = []
    files_names = []
    for file in os.listdir(folder):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(folder, file)

            reader = PdfReader(pdf_path)
            paragraph = ''.join([page.extract_text() for page in reader.pages])
            
            company.append(paragraph)
            files_names.append(file)
    return company, files_names
