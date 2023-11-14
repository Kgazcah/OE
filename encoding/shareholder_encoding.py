import torch
from collections import OrderedDict
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer

class ShareholderEmbeddings():
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased',
                output_hidden_states = True,)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def whitespace_tokenize(self, marked_text):
        """Runs basic whitespace splitting on a piece of text."""
        if not marked_text:
            return []
        self.tokenized_text = marked_text.split()

    def bert_text_preparation(self, sentence):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        marked_text = "[CLS] " + sentence + " [SEP]"
        self.whitespace_tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenized_text)
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

        # concatenate the tensors for all layers
        # use "stack" to create new dimension in tensor
        token_embeddings = torch.stack(hidden_states, dim=0)

        # remove dimension 1, the "batches"
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # swap dimensions 0 and 1 so we can loop over tokens
        token_embeddings = token_embeddings.permute(1,0,2)

        # intialized list to store embeddings
        token_vecs_sum = []

        # "token_embeddings" is a [Y x 12 x 768] tensor
        # where Y is the number of tokens in the sentence, 12 is the layers in BERT model & 768 the embedding dimension in each layer

        # loop over tokens in sentence
        for token in token_embeddings:

            # "token" is a [12 x 768] tensor
            # sum the vectors from the last four layers
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        return token_vecs_sum
    

    def getting_token_embeddings(self, sentences):
        self.sentences = sentences
        self.context_embeddings = []
        self.context_tokens = []

        for sentence in self.sentences:
            self.bert_text_preparation(sentence)
            list_token_embeddings = self.get_bert_embeddings()

            # make ordered dictionary to keep track of the position of each word
            tokens = OrderedDict()

            # loop over tokens in sensitive sentence
            for token in self.tokenized_text[1:-1]:
            # keep track of position of word and whether it occurs multiple times
                if token in tokens:
                    tokens[token] += 1
                else:
                    tokens[token] = 1

                # compute the position of the current token
                token_indices = [i for i, t in enumerate(self.tokenized_text) if t == token]
                current_index = token_indices[tokens[token]-1]

                # get the corresponding embedding
                token_vec = list_token_embeddings[current_index]

                # save values
                self.context_tokens.append(token)
                self.context_embeddings.append(token_vec.numpy())
    

    def save_token_embeddings(self,  save_words_as, save_embeddings_as):
        #dataframe transformation
        context_tokens_dataframe = pd.DataFrame(self.context_tokens)
        context_embeddings_dataframe = pd.DataFrame(self.context_embeddings)

        #converting and saving in tsv
        context_tokens_dataframe.to_csv(save_words_as+".tsv", sep="\t", index=False, header=False)
        context_embeddings_dataframe.to_csv(save_embeddings_as+".tsv", sep="\t", index=False, header=False)