import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig

class SentenceClassification_Dataset(Dataset):

    def __init__(self, text, lang):
        super(SentenceClassification_Dataset, self).__init__() 
        self.text = text
        self.lang = lang
        self.lang_id = {_lang:idx for idx, _lang in enumerate(sorted(list(set(self.lang))))}

    @classmethod
    def load_data(cls, infile):
        dataset = load_dataset('json', data_files=infile)
        text = dataset['train']['text']
        lang = dataset['train']['lang']    
        return cls(text, lang)
    
    def __len__(self):
        assert len(self.text) == len(self.lang)
        return len(self.text)
    
    def __getitem__(self, index):
        text = self.text[index]
        lang = self.lang_id[self.lang[index]]
        return {'text': text,
                'lang': lang} 

class TokenDetection_Dataset(object):
    
    def __init__(self, dataset):
        super(TokenDetection_Dataset, self).__init__() 
        self.dataset = dataset
                
    @classmethod
    def load_data(cls, infile, tokenizer):
        dataset = load_dataset('json', data_files=infile)
        dataset = dataset.map(lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True)
        processed_dataset = dataset.remove_columns(['text', 'label', 'en', 'multi'])
        return cls(processed_dataset['train'])

class Sentence_Collator(object):
    
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.padding = args.padding      
        self.max_length = args.max_length
        self.truncation = args.truncation

    def __call__(self, samples):
        text_lst, lang_lst = [], []
        for sample in samples:
            text_lst.append(sample['text'])
            lang_lst.append(sample['lang'])

        text_encode = self.tokenizer(
            text_lst,
            padding = self.padding,
            max_length = self.max_length,
            truncation = self.truncation,
            )
    
        batch = {
            'input_ids':torch.tensor(text_encode['input_ids']),
            'attention_mask':torch.tensor(text_encode['attention_mask']),
            'token_type_ids':torch.tensor(text_encode['token_type_ids']),        
            'label': torch.tensor(lang_lst)
        }        
        return batch
    
def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["text"], padding=True, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['label']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs