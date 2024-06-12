import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class Encoder(nn.Module):
    
    def __init__(self, args):
        super(Encoder, self).__init__()        
        self.config = AutoConfig.from_pretrained(args.model_dir)
        self.model = AutoModel.from_pretrained(args.model_dir, return_dict =True, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, config=args.model_dir)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.num_layer = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        self.probe_type = args.probe_type
    
    def forward(self, input_ids, attention_mask, token_type_ids, probe_layer, pooler=None): 
        outputs =  self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        assert 0 < probe_layer <= self.num_layer
        
        #### cumulative scoring ####
        if self.probe_type == 'cumulative_scoring':
            if pooler == 'cls':
                hidden_states = [outputs.hidden_states[i][:,0,:] for i in range(1, probe_layer+1)]
            elif pooler == 'avg':
                hidden_states = []
                for i in range(1, probe_layer+1):
                    avg_pooled = ((outputs.hidden_states[i] * attention_mask.unsqueeze(-1)).sum(1) \
                                  / attention_mask.sum(-1).unsqueeze(-1))
                    hidden_states.append(avg_pooled)
            else:
                # for token detection task
                hidden_states = [outputs.hidden_states[i] for i in range(1, probe_layer+1)]
            return sum(hidden_states)

        #### layer-wise probing ####
        else:
            if pooler == 'cls':
                hidden_states = outputs.hidden_states[probe_layer][:,0,:]
            elif pooler == 'avg':
                hidden_states = ((outputs.hidden_states[probe_layer] * attention_mask.unsqueeze(-1)).sum(1) \
                                 / attention_mask.sum(-1).unsqueeze(-1))
            else:
                # for token detection task
                hidden_states = outputs.hidden_states[probe_layer]
            return hidden_states            


class Sentence_Classifier(nn.Module):
    
    def __init__(self, hidden_size, num_lang):
        super(Sentence_Classifier, self).__init__()
        self.head = nn.Linear(hidden_size, num_lang)
        self.tanh = nn.Tanh() 

    def forward(self, feature):
        x = self.tanh(feature)
        output = self.head(x)
        return output

class TokenDetection_Classifier(nn.Module):
    
    def __init__(self, hidden_size):
        super(TokenDetection_Classifier, self).__init__()
        self.head = torch.nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feature):
        x = self.head(feature)
        output = self.sigmoid(x)
        return output