from preprocessing.shareholder_preprocessing import ShareholderPreprocessing
from similarity.shareholder_similarity import ShareholderSimilarity
from encoding.shareholder_encoding import ShareholderEmbeddings
import files_name.text as text
import pandas as pd
import os
import re

########## getting the dataframe ############################
preprocessing = ShareholderPreprocessing()
folder_input = 'empresas/shareholders'
folder_output = 'empresas'
files_names = text.file_name_list(folder_input,".txt")

for file_name in files_names:
    with open(f'{folder_input}/{file_name}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        
    sentences = preprocessing.lowering_case(text)
    sentences = preprocessing.removing_url(sentences)
    sentences = preprocessing.removing_numbers(sentences)
    sentences = preprocessing.removing_punctuation(sentences)
    sentences = preprocessing.some_custom_stopwords(sentences)
    sentences = preprocessing.removing_title(sentences)
    sentences = preprocessing.cleaning_points(sentences)
    sentences = sentences.split('.')
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]

    df = pd.DataFrame(sentences)
    new_folder = f'{folder_output}/{file_name}'
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        print(f'La carpeta {new_folder} ya existe.')
    except Exception as e:
        print(f'Error al crear la carpeta: {e}')
    df.to_csv(f'{new_folder}/{file_name}_words.tsv', sep='\t', header=False, index=False)
   
################# encoding #######################
encoding = ShareholderEmbeddings()

for company_shareholder in os.listdir(folder_output):
    if company_shareholder != 'shareholders':
        subfolder_path = os.path.join(folder_output, company_shareholder)
        
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            words_pattern = re.compile(r'.*words\.tsv')
            words_files = list(filter(words_pattern.match, files))

            if words_files:
                words_path = os.path.join(subfolder_path, words_files[0])
                shareholder_words = pd.read_csv(words_path, sep='\t', header=None)
                sentences = shareholder_words[0].to_numpy(dtype=str)
                if pd.isna(sentences[-1]):
                    sentences=sentences[:-1]
                encoding.getting_sentence_embeddings(sentences)
                encoding.save_token_embeddings(f"{folder_output}/{company_shareholder}/{company_shareholder}_embeddings")
                

############## similarity ##############
comparing = ShareholderSimilarity()

centroids_embeddings_path = 'data/centroids_embeddings.tsv'
centroids_words_path = 'data/centroids_words.tsv'
porcentages = 'porcentages.csv'
similarity_document = 'similarities.csv'
column_names = ['company','year','autonomy','competitive agresiveness','innovativeness','proactiveness','risk taking']

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
                
                shareholder_words = pd.read_csv(words_path, header=None)
                shareholder_embeddings = pd.read_csv(embeddings_path, sep='\t', header=None)
                
                comparing.similarity(centroids_words, centroids_embeddings, shareholder_words, shareholder_embeddings,subfolder_path)
                comparing.calculate_porcentage(subfolder_path, similarity_document, column_names, porcentages)