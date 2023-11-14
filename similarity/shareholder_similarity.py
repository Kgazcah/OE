from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

class ShareholderSimilarity:
    def __init__(self):
        pass

    def similarity(self, centroids_words, centroids_embeddings, shareholder_words, shareholder_embeddings, output_path):
        shareholder_rows = shareholder_embeddings.shape[0]
        centroids_rows = centroids_embeddings.shape[0]
        matrix = np.zeros((shareholder_rows, centroids_rows))
        centroids_classes = centroids_words.values.flatten().tolist()

        sim_classes = []

        for i in range(shareholder_rows):
            shareholder_vector = shareholder_embeddings.iloc[i, :].values
            similarities = []
            for j in range(centroids_rows):
                centroid_vector = centroids_embeddings.iloc[j, :].values
                sim = self.calculate_similarity([shareholder_vector], [centroid_vector])
                similarities.append(sim[0][0])

            sim_class = similarities.index(max(similarities)) + 1  
            sim_classes.append(sim_class)

            matrix[i, :] = similarities

        similarities_df = pd.DataFrame(matrix, index=range(shareholder_rows), columns=centroids_classes)
        similarities_df['Class'] = sim_classes

        # Agregar la columna de palabras de shareholder_words al inicio del DataFrame
        similarities_df.insert(0, 'Words', shareholder_words.values)
        similarities_df.to_csv(f'{output_path}/similarities.csv', index=False)


    def calculate_similarity(self, shareholder_vector, centroid_vector):
        # Calcular la similitud del coseno entre el vector de referencia y los vectores de comparación.
        self.similarities = cosine_similarity(shareholder_vector, centroid_vector)
        return self.similarities

    def calculate_porcentage(self, subfolder_path, similarity_document, column_names, porcentages):
        try:
            df_porcentages = pd.read_csv(porcentages)
        except FileNotFoundError:
            df_porcentages = pd.DataFrame(columns=column_names)

        t_df_porcentages = pd.DataFrame(columns=column_names)

        sim_document = [pd.read_csv(f"{subfolder_path}/{similarity_document}")]
        for i, sd in enumerate(sim_document):
            t_rows = len(sim_document[0])
            porcentage_class1 = round((sd['Class'] == 1).sum() / t_rows,4)
            porcentage_class2 = round((sd['Class'] == 2).sum() / t_rows,4)
            porcentage_class3 = round((sd['Class'] == 3).sum() / t_rows,4)
            porcentage_class4 = round((sd['Class'] == 4).sum() / t_rows,4)
            porcentage_class5 = round((sd['Class'] == 5).sum() / t_rows,4)


        cn_g = [match.group() for match in (re.search(r'\b\w+\b', c) for c in str(subfolder_path).split()) if match]
        company_name = next(iter(cn_g), None)
        year_g = [int(match.group()) for match in (re.search(r'\b\d+\b', y) for y in str(subfolder_path).split()) if match]
        year = next(iter(year_g), None)

        new_row = {column_names[0]:company_name,column_names[1]:year,column_names[2]:porcentage_class1,column_names[3]:porcentage_class2,
                    column_names[4]:porcentage_class3,column_names[5]:porcentage_class4,column_names[6]:porcentage_class5}
        
        t_df_porcentages = pd.concat([t_df_porcentages, pd.DataFrame([new_row])], ignore_index=True)


        f_df_porcentages = pd.concat([df_porcentages, t_df_porcentages], ignore_index=True)
        f_df_porcentages.to_csv(porcentages, index=False)






