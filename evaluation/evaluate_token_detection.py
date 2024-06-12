import os
import re
import time
import random
import datetime
import logging
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from transformers import DataCollatorForTokenClassification, AutoTokenizer

from utils import (
    print_table,
    save_result,
    calculate_differences,
    extract_layer_number,
    compute_mean_vector,
    compute_principal_component,
)

import sys
sys.path.append("../") 

from src.data_loader import TokenDetection_Dataset, Sentence_Collator
from src.probe_model import Encoder, TokenDetection_Classifier

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='sentence identification evaluation')

    # Required
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory for pretrained model'
                       )    
    parser.add_argument('--pt_dir', type=str, required=True,
                        help='Directory for .pt files of probing models'
                       )        
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test set directory'
                       )
    parser.add_argument('--output_dir', type=str, default='../result',
                        help='Directory for output'
                       )
    parser.add_argument('--probe_type', type=str, required=True,
                        help='Choose probe type: {cumulative_scoring|layer_wise}'
                       )

    # Tokenizer & Collator settings
    parser.add_argument('--max_length', default=256, type=int,
                        help='Max length of sequence'
                       )
    parser.add_argument('--padding', action="store_true",
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_true",
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_true",
                        help='Load shuffled sequences'
                       )
    
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed'
                       )
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                       help='Choose device for evaluatoin'
                       )
      
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss  

def get_f1_score(logits, labels, attention_mask):
    logits = logits.argmax(-1) 
    
    y_true = labels[attention_mask == 1]
    y_pred = logits[attention_mask == 1]

    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def evaluate(encoder, probe_model, test_dataloader, probe_layer, args, identifier=None):
    f1_score_lst = []
    probe_model.eval()
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
        
            # pass the data to device(cpu or gpu)
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)   

            # hidden_states_summed : (batch_size, hidden_size)
            hidden_states_summed = encoder(input_ids, attention_mask, token_type_ids, probe_layer)                                   
            logits = probe_model(hidden_states_summed)
        
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            attention_mask = attention_mask.to('cpu').numpy()
  
            f1_score = get_f1_score(logits, labels, attention_mask)
            f1_score_lst.append(f1_score)

        f1_score = sum(f1_score_lst)/len(f1_score_lst)
    #print('Accuracy: {:.2f}(%)'.format(accuracy))
    return f1_score

def main(args):
    init_logging()
    seed_everything(args)
    
    LOGGER.info('*** Token Detection Evaluation ***')
    LOGGER.info('Probe Type: {}'.format(args.probe_type))   
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    test_dataset = TokenDetection_Dataset.load_data(args.test_dir, tokenizer)
    
    collator = DataCollatorForTokenClassification(tokenizer)
    
    test_dataloader = DataLoader(test_dataset.dataset,
                                 batch_size=args.batch_size,
                                 shuffle=args.shuffle,
                                 collate_fn=collator)
    
    pt_file_path = []
    for file in os.listdir(args.pt_dir):
        if args.probe_type == 'cumulative_scoring':
            if 'token_detection_cumulative_scoring' in file:
                path = args.pt_dir + '/' + file
                pt_file_path.append(path)
        else:
            if 'token_detection_layer_wise' in file:
                path = args.pt_dir + '/' + file
                pt_file_path.append(path)
    
    assert len(pt_file_path) > 0, 'No .pt files found in the specified path.'
    
    pt_file_path = sorted(pt_file_path, key=extract_layer_number)

    layers_lst = []
    result_lst = []
    for pt in pt_file_path:
        encoder = Encoder(args).to(args.device)
        #probe_model = Sentence_Classifier(encoder.hidden_size, len(test_dataset.lang_id)).to(args.device)
        #probe_model.load_state_dict(torch.load(pt))
        probe_model = torch.load(pt)

        #for name, param in encoder.named_parameters():
        #    param.requires_grad = False
        
        # checkpoint example : token_detection_layer-1-1_epoch-15.pt
        probe_layer = int(extract_layer_number(pt))
        
        result = evaluate(encoder, probe_model, test_dataloader, probe_layer, args)   
        
        result_lst.append("%.2f" % result)
        layers_lst.append("%d" % probe_layer)
    
    if args.probe_type == 'cumulative_scoring':
        diff_lst = calculate_differences(result_lst)   
        print('*** Score Change ***')
        print_table(layers_lst, diff_lst)
        save_result(args.output_dir, 'score_change.txt', layers_lst, diff_lst)
    
        print('*** Score ***')
        layers_lst = ['1-'+str(layer) for layer in layers_lst]
        layers_lst.append("Avg.")
        result_lst.append("%.2f" % (sum([float(result) for result in result_lst]) / len(result_lst)))

    else:
        print('*** Score ***')
        #layers_lst = ['1-'+str(layer) for layer in layers_lst]
        layers_lst.append("Avg.")
        result_lst.append("%.2f" % (sum([float(result) for result in result_lst]) / len(result_lst)))        
    
    print_table(layers_lst, result_lst)
    save_result(args.output_dir, 'score.txt', layers_lst, result_lst)
    
if __name__ == '__main__':
    args = argument_parser()
    main(args)