import torch
import pandas as pd
from transformers import BertModel, BertTokenizer

class ShareholderEmbeddings():
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def bert_text_preparation(self, sentence):
        marked_text = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        if len(tokenized_text) > 512:
            self.tokens_tensor, self.segments_tensor = self.process_and_get_tensors(tokenized_text)
        else:         
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(indexed_tokens)

            # convert inputs to tensors
            self.tokens_tensor = torch.tensor([indexed_tokens])
            self.segments_tensor = torch.tensor([segments_ids])

    def get_bert_embeddings(self):
        if isinstance(self.tokens_tensor, list):
            values = None
            tokens_tensors = self.tokens_tensor
            segments_tensors = self.segments_tensor
            for self.tokens_tensor, self.segments_tensor in zip(tokens_tensors, segments_tensors):
                if values is None:
                    values = self.get_bert_embeddings()
                else:
                    values += self.get_bert_embeddings()
            return values / len(tokens_tensors)
        else:
            with torch.no_grad():
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
        context_embeddings_dataframe = pd.DataFrame(self.context_embeddings)
        context_embeddings_dataframe.to_csv(save_embeddings_as + ".tsv", sep='\t', index=False, header=False)

    def process_and_get_tensors(self, tokens, chunk_size=512):

        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        tokens_tensor, segments_tensor = [], []
        for chunk_tokens in chunks:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
            segments_ids = [1] * len(indexed_tokens)

            tokens_tensor.append(torch.tensor([indexed_tokens]))
            segments_tensor.append(torch.tensor([segments_ids]))

        return tokens_tensor, segments_tensor
