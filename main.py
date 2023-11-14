from preprocessing.shareholder_preprocessing import ShareholderPreprocessing
from encoding.shareholder_encoding import ShareholderEmbeddings
from similarity.shareholder_similarity import ShareholderSimilarity
import numpy as np
import pandas as pd
import pdf_to_text.text as text
import os
import re


preprocessing = ShareholderPreprocessing()
encoding = ShareholderEmbeddings()
comparing = ShareholderSimilarity()

folder_input = 'empresa2/shareholders'
folder_output = 'empresa2/'
centroids_embeddings_path = 'data/centroids_embeddings.tsv'
centroids_words_path = 'data/centroids_words.tsv'
porcentages = 'porcentages.csv'
similarity_document = 'similarities.csv'
column_names = ['company','year','autonomy','competitive agresiveness','innovativeness','proactiveness','risk taking']

company, files_names = text.pdf_text(folder_input)

#removing .pdf from the file name 
for i in range(len(files_names)):
    files_names[i] = files_names[i].replace(".pdf", "")


for (paragraph, file_name) in zip(company, files_names):
    output_folder = os.path.join(folder_output,file_name)
    os.makedirs(output_folder, exist_ok=True)

    sentences = [sentence.strip() for sentence in paragraph.split('. ')]
    sentences = [preprocessing.cleaning_sentence(sentence) for sentence in sentences]
    sentences = [preprocessing.lowering_case(sentence) for sentence in sentences]
    sentences = [preprocessing.removing_stopwords(sentence) for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence != '']

    encoding.getting_token_embeddings(sentences)
    encoding.save_token_embeddings(os.path.join(output_folder, f"{file_name}_words"), os.path.join(output_folder, f"{file_name}_embeddings"))

centroids_embeddings = pd.read_csv(centroids_embeddings_path, sep='\t', header=None)
centroids_words = pd.read_csv(centroids_words_path, sep='\t', header=None)
                 
for company_shareholder in os.listdir(folder_output):
    if company_shareholder != 'shareholders':
        subfolder_path = os.path.join(folder_output, company_shareholder)
        
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            
            words_pattern = re.compile(r'.*words\.tsv')
            embeddings_pattern = re.compile(r'.*embeddings\.tsv')
            words_files = list(filter(words_pattern.match, files))
            embeddings_files = list(filter(embeddings_pattern.match, files))

            if words_files and embeddings_files:
                words_path = os.path.join(subfolder_path, words_files[0])
                embeddings_path = os.path.join(subfolder_path, embeddings_files[0])
                
                shareholder_words = pd.read_csv(words_path, sep='\t', header=None)
                shareholder_embeddings = pd.read_csv(embeddings_path, sep='\t', header=None)
                comparing.similarity(centroids_words, centroids_embeddings, shareholder_words, shareholder_embeddings,subfolder_path)
                comparing.calculate_porcentage(subfolder_path, similarity_document, column_names, porcentages)
                


