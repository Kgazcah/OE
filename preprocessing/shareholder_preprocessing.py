import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
custom_stopwords = set(stopwords.words('english'))

# Adding custom stopwords
custom_stopwords.update(['inc', 'per', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                        'october', 'november', 'december', 'add', 'httpswww', 'www', 'httpwww'])

class ShareholderPreprocessing():
   
    #cleaning what is not letters from the sentence
    def cleaning_sentence(self,sentence):
        clean_sentence = ''.join(s for s in sentence if s.isalpha() or s.isspace())
        return clean_sentence

    # Lowering case
    def lowering_case(self,sentence):
        clean_sentence = ''.join(s.lower() for s in sentence)
        return clean_sentence

    # Removing stopwords
    def removing_stopwords(self,sentence):
        words = sentence.split()
        filtered_words = [word for word in words if len(word) >= 3]
        clean_words = [word for word in filtered_words if word.lower() not in custom_stopwords]
        clean_sentence = ' '.join(clean_words)
        return clean_sentence
    

