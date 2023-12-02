import re

custom_stopwords = ['item','httpswww', 'www', 'httpwww', 'inc']

class ShareholderPreprocessing():
   
    def removing_url(self,text):
        no_url_text = re.sub(r'https://\S+', '', text)
        no_url_text = re.sub(r'www\.\w+\.com', '', no_url_text)
        no_url_text = re.sub(r'\.\w+\.\w*\/*\w*\/*\w*\/*\w*\/*\d*\/*[0-9a-zA-Z]*\-*\d*\/*[a-zA-Z]*\-*\d*\.*htm*', '', no_url_text)
        return no_url_text
    
    def removing_title(self,text):
        no_title = text.replace("managements discussion and analysis of financial condition and results of operations", '')
        return no_title
    
    def removing_numbers(self,text):
        sentences = re.sub(r'\d+[a-z]+','',text)
        sentences = re.sub(r'\d+\.\d*','',sentences)
        sentences = re.sub(r'[a-z]+\d+','',sentences)
        no_numbers_text = re.sub(r'\$+\d+\.*\d*' ,'', sentences)
        no_numbers_text = re.sub(r'\b\d+\b', '', no_numbers_text)
        return no_numbers_text

    def lowering_case(self,text):
        lower_case_text = text.lower()
        return lower_case_text

    def removing_punctuation(self,text):
        """
        Remove any character except the point
        """
        
        no_punctuation = re.sub(r'[^\w\s.\']', '', text)
        no_punctuation = re.sub(r'\'+\s+','s ',no_punctuation)
        
        return no_punctuation
    
    def cleaning_points(self,text):
        """
        Remove two points in a row
        """
        no_points_in_a_row = re.sub(r'\.\s*\.', '.' ,text)
        return no_points_in_a_row
    
    def some_custom_stopwords(self,text):
        words = ['table of contents','item','form k', 'aig', '½', 'ltd.', 'inc.', 'part']
        for word in words:
            text = text.replace(f'{word}', '')
        sentences = re.sub(r'\s+i+\s+','',text)
        sentences = re.sub(r'[a-z]+\.+[a-z]+\.*[a-z]*', '', sentences)
        sentences = re.sub(r'\.+\s*[a-z]+\s*\-*\.+', '.', sentences)
        sentences = re.sub(r'\d+\_+\s*\.*\s*\w*\.*', '', sentences)
        sentences = re.sub(r'\_*htm', '', sentences)
        sentences = re.sub(r'\s*u\.s\.a*\s*',' usa ', sentences)
        sentences = re.sub(r'\s*s\.a\.\s+',' ', sentences)
        sentences = re.sub(r'\s*vs\.\s+',' vs ', sentences)
        sentences = re.sub(r'\s*u\.k\.\s+',' uk ', sentences)
        sentences = re.sub(r'\s*d\.c\.\s+',' dc ', sentences)
        sentences = re.sub(r'\s+s\s+',' ', sentences)
        sentences = re.sub(r'\s+iv\s+',' ', sentences)
        sentences = re.sub(r'\s+vi*\s+','', sentences)
        sentences = re.sub(r'\s+i\.e\.\s+', ' ', sentences)
        sentences = re.sub(r'\s+e\.g\.\s+', ' ', sentences)
        sentences = re.sub(r'\s+a\.m\.\s+', ' ', sentences)
        sentences = re.sub(r'\s+p\.m\.\s+', ' ', sentences)
        sentences = re.sub(r'\n',' ', sentences)
        sentences = re.sub(r'\s+',' ', sentences)
        sentences = re.sub(r'(\.\s+\b\w+\b\s+\.)','.', sentences)
        sentences = re.sub(r'(\.\s+(\b\w+\b)\s+(\b\w+\b)\s*\.)','.', sentences)
        sentences = re.sub(r't\s*a\s*b\s*l\s*e\s*o\s*f\s*c\s*o\s*n\s*t\s*e\s*n\s*t\s*s*','', sentences)
        sentences = re.sub(r'\s+[a-z]{1}\s+',' ', sentences)
        sentences = re.sub(r'\.\s+','.', sentences)
        return sentences

    


