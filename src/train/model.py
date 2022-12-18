from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, pretrained_model):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        
    def forward(self, **encoded_input):
        
        # Normal model forward
        model_output = self.model(**encoded_input)
        
        # Get sentence embeddings
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings