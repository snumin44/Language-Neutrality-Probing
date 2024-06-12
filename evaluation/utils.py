import re
import os
from prettytable import PrettyTable
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

def print_table(layer_nums, scores):
    tb = PrettyTable()
    tb.field_names = layer_nums
    tb.add_row(scores)
    print(tb)

def save_result(output_path, file_name, layer_nums, scores):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_name = os.path.join(output_path, file_name)
    with open(file_name, 'w') as file:
        for layer, score in zip(layer_nums, scores):
            file.write(str(layer) + '\t' + str(score) + '\n')

def calculate_differences(results):
    differences = [0] 
    for i in range(1, len(results)):
        difference = float(results[i]) - float(results[i - 1])
        differences.append(round(difference, 3))
    return differences

def extract_layer_number(file_name):
    match = re.search(r'layer-1-(\d+)_|layer-(\d+)_', file_name)
    if match:
        return int(match.group(1) or match.group(2))
    else:
        return float('inf') 

def compute_mean_vector(encoder, dataset, probe_layer, pooler, args):
    # seperate text by each language
    lang_set = {}
    for idx, text in zip(dataset.lang_id.values(), dataset.text):
        if idx in lang_set.keys():
            texts = lang_set[idx]
            texts.append(text)
        else:
            lang_set[idx] = [text]
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # compute mean vector for each language
    mean_vectors = {}
    for idx in tqdm(lang_set.keys()):
        sentences = lang_set[idx] 

        vectors = []
        for start_index in range(0, len(sentences), args.batch_size):
            # Divide Sentences into Mini-Batch
            batch = sentences[start_index : start_index + args.batch_size]
           
            features = tokenizer(batch,
                                padding = args.padding,
                                max_length = args.max_length,
                                truncation = args.truncation,)
            
            encoder.eval()
            with torch.no_grad():
                hidden_states = encoder(input_ids = torch.tensor(features['input_ids']).to(args.device),
                                        attention_mask = torch.tensor(features['attention_mask']).to(args.device),
                                        token_type_ids = torch.tensor(features['token_type_ids']).to(args.device),
                                        probe_layer = probe_layer,
                                        pooler = pooler,
                                        )
            vectors.append(hidden_states)
        mean_vector = torch.concat(vectors, dim=0).mean(0)
        mean_vectors[idx] = mean_vector            
                
    return mean_vectors 

def compute_principal_component(encoder, dataset, probe_layer, pooler, args):
    # seperate text by each language
    lang_set = {}
    for idx, text in zip(dataset.lang_id.values(), dataset.text):
        if idx in lang_set.keys():
            texts = lang_set[idx]
            texts.append(text)
        else:
            lang_set[idx] = [text]
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # compute principal component for each language
    principal_components = {}
    for idx in tqdm(lang_set.keys()):
        sentences = lang_set[idx] 

        vectors = []
        for start_index in range(0, len(sentences), args.batch_size):
            # Divide Sentences into Mini-Batch
            batch = sentences[start_index : start_index + args.batch_size]
           
            features = tokenizer(batch,
                                padding = args.padding,
                                max_length = args.max_length,
                                truncation = args.truncation,)
            
            encoder.eval()
            with torch.no_grad():
                hidden_states = encoder(input_ids = torch.tensor(features['input_ids']).to(args.device),
                                        attention_mask = torch.tensor(features['attention_mask']).to(args.device),
                                        token_type_ids = torch.tensor(features['token_type_ids']).to(args.device),
                                        probe_layer = probe_layer,
                                        pooler = pooler,
                                              )
            vectors.append(hidden_states)
        total_vector = torch.concat(vectors, dim=0) # (total_num, hidden_size)
        
        # compute principal component with SVD 
        # Note that full matrices = False. U : (m x k), S : (k=min(m,n)), V : (k x n)  
        U, S, V = torch.linalg.svd(total_vector, full_matrices=False) 
        principal_component = V[0]
        principal_components[idx] = principal_component           
                
    return principal_components 