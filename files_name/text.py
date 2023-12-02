import os

def file_name_list(folder,ext):
    files_in_folder = os.listdir(folder)
    txt_files = [file for file in files_in_folder if file.endswith(ext)]
    files_names = [os.path.splitext(archivo)[0] for archivo in txt_files]
    return files_names