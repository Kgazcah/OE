import torch
import pandas as pd
from transformers import BertModel, BertTokenizer

class ShareholderEmbeddings():
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased',
                output_hidden_states = True,)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def bert_text_preparation(self, sentence):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)

        # convert inputs to tensors
        self.tokens_tensor = torch.tensor([indexed_tokens])
        self.segments_tensor = torch.tensor([segments_ids])
    
    def get_bert_embeddings(self):
        """
        Obtains BERT embeddings for tokens, in context of the given sentence.
        """
        # gradient calculation id disabled
        with torch.no_grad():
            # obtain hidden states
            outputs = self.model(self.tokens_tensor, self.segments_tensor)
            hidden_states = outputs[2]
        
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        sentence_embedding = torch.sum(token_embeddings[-4:], dim=0)
        return sentence_embedding.numpy()
    

    def getting_sentence_embeddings(self, sentences):
        self.sentences = sentences
        self.context_embeddings = []

        for sentence in self.sentences:
            self.bert_text_preparation(sentence)
            embedding = self.get_bert_embeddings()
            self.context_embeddings.append(embedding[0])
    

    def save_token_embeddings(self, save_embeddings_as):
        #dataframe transformation
        context_embeddings_dataframe = pd.DataFrame(self.context_embeddings)

        #converting and saving in tsv
        context_embeddings_dataframe.to_csv(save_embeddings_as+".tsv", sep='\t', index=False, header=False)

