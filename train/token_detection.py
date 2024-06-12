import os
import re
import time
import random
import datetime
import logging
import numpy as np
import pandas as pd
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForTokenClassification, AutoTokenizer

import sys
sys.path.append("../") 

from src.data_loader import TokenDetection_Dataset
from src.probe_model import Encoder, TokenDetection_Classifier

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='token detection train')

    # Required
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory for pretrained model'
                       )    
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Training set directory'
                       )
    parser.add_argument('--valid_dir', type=str, required=True,
                        help='Validation set directory'
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

    # Train config    
    parser.add_argument('--epochs', default=15, type=int,
                        help='Training epochs'
                       )  
    parser.add_argument('--probe_layer', default=12, type=int,
                        help='Train a classifier for specific layer.'
                       )       
    parser.add_argument('--early_stop', default=5, type=int,
                        help='Number of epochs for early stopping'
                       )
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay'
                       )       
    parser.add_argument('--learning_rate', default=4e-4, type=float,
                        help='Leraning rate'
                       )       
    parser.add_argument('--gamma', default=0.996, type=float,
                        help='Gamma for schuedler'
                       )   
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon for schuedler'
                       )   
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
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

    #rep = (labels == 1) * attention_mask
    #fix = (labels == 0) * attention_mask
    #rep_acc = float(((prediction == 1) * rep).sum() / rep.sum())
    #fix_acc = float(((prediction == 0) * fix).sum() / fix.sum())
    #rep_acc = float((prediction * rep).sum() / rep.sum())
    #fix_acc = float(1.0 - (prediction*fix).sum() / fix.sum())
    #acc = float(((prediction == labels) * attention_mask).sum() / attention_mask.sum()) ### 이게 accuracy 
    #return acc, rep_acc, fix_acc

def get_adamw_optimizer(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, weight_decay=args.weight_decay, eps = args.eps)
    return optimizer

def get_explr_scheduler(optimizer, args):
    scheduler = ExponentialLR(optimizer, gamma = args.gamma, last_epoch=-1,verbose=False)
    return scheduler 

def train(encoder, probe_model, train_dataloader, valid_dataloader, optimizer, scheduler, args):
    epoch_step = 0
    early_stop_loss = list()
    total_t0 = time.time()
    best_loss, best_model = None, None
    
    for epoch_i in range(args.epochs):           
        
        ######## TRAINING LOOP ########
        t0 = time.time()
        
        total_train_loss = 0
                
        probe_model.train()
        for _, batch in enumerate(train_dataloader):
            
            # pass the data to device(cpu or gpu)
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)

            probe_model.zero_grad()

            hidden_states_summed = encoder(input_ids, attention_mask, token_type_ids, args.probe_layer)
            logits = probe_model(hidden_states_summed)
            
            # Pytorch Cross-Entropy function includes the function of one-hot encoding and softmax.
            train_loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
            total_train_loss += train_loss.item()
            
            train_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=5.0)            
            
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        
        training_time = format_time(time.time() - t0)
        
        print("[epoch {}] average train loss    :{}".format(epoch_i+1, avg_train_loss)) 
        print("[epoch {}] training time         :{}".format(epoch_i+1, training_time))

        ######## VALIDATION LOOP ########
        t0 = time.time()
    
        total_valid_loss = 0
        total_valid_f1 = 0
        #total_valid_rep_acc = 0 
        #total_valid_fix_acc = 0
        
        encoder.eval()
        probe_model.eval()    
        for _, batch in enumerate(valid_dataloader):
            
            # pass the data to device(cpu or gpu)
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction.
            with torch.no_grad():
                hidden_states_summed = encoder(input_ids, attention_mask, token_type_ids, args.probe_layer)
                logits = probe_model(hidden_states_summed)
                
                valid_loss =  F.cross_entropy(logits.view(-1, 2), labels.view(-1))
                total_valid_loss += valid_loss.item()
                                
            # You have to send the result from GPU to CPU in order to conduct calculation with numpy.             
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            attention_mask = attention_mask.to('cpu').numpy()
            
            #valid_acc, valid_rep_acc, valid_fix_acc = get_accuracy(logits, labels, attention_mask)    
            valid_f1 = get_f1_score(logits, labels, attention_mask)
            total_valid_f1 += valid_f1
            #total_valid_rep_acc += valid_rep_acc
            #total_valid_fix_acc += valid_fix_acc

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        avg_valid_f1 = total_valid_f1 / len(valid_dataloader)
        #avg_valid_rep_acc = total_valid_rep_acc / len(valid_dataloader)
        #avg_valid_fix_acc = total_valid_fix_acc / len(valid_dataloader)            
        
        # Update the optimal model with the smallest validation error and loss & epoch information.
        if not best_loss or avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_model = deepcopy(probe_model)
            epoch_step = epoch_i + 1
    
        valididation_time = format_time(time.time() - t0)
        
        print("[epoch {}] average valid loss    :{}".format(epoch_i+1, avg_valid_loss))
        print("[epoch {}] average valid f1-score:{}".format(epoch_i+1, avg_valid_f1))        
        #print("[epoch {}] average rep accuracy  :{}".format(epoch_i+1, avg_valid_rep_acc)) 
        #print("[epoch {}] average fix accuracy  :{}".format(epoch_i+1, avg_valid_fix_acc)) 
        print("[epoch {}] validation time       :{}\n".format(epoch_i+1, valididation_time))       
                    
        # When valid_loss is greater than the previous valid_loss, add it to the early_stop_loss list. 
        # Stop training when the length of the list reaches the early_stop set by the user.   
        if len(early_stop_loss) == 0 or avg_valid_loss > early_stop_loss[-1]:
            early_stop_loss.append(avg_valid_loss)
            if len(early_stop_loss) == args.early_stop:break                                      
        else: early_stop_loss = list()

    # Save the best model.
    encoder_name = args.model_dir.split('/')[-1]
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.probe_type == 'cumulative_scoring':
        model_path = os.path.join(output_dir,
                                  "token_detection_{}_layer-1-{}_epoch-{}.pt".format(str(args.probe_type),
                                                                                     str(args.probe_layer),
                                                                                     str(epoch_step)))   
    else:
        model_path = os.path.join(output_dir,
                                  "token_detection_{}_layer-{}_epoch-{}.pt".format(str(args.probe_type),
                                                                                   str(args.probe_layer),
                                                                                   str(epoch_step)))       
    

    torch.save(best_model, model_path)
    
    print('TRAIN DONE')
    print("Total training took {:} (h:mm:ss)\n".format(format_time(time.time()-total_t0)))


def main(args):
    init_logging()
    seed_everything(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    train_dataset = TokenDetection_Dataset.load_data(args.train_dir, tokenizer)
    valid_dataset = TokenDetection_Dataset.load_data(args.valid_dir, tokenizer)

    collator = DataCollatorForTokenClassification(tokenizer)
    
    train_dataloader = DataLoader(train_dataset.dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  collate_fn=collator)
    
    valid_dataloader = DataLoader(valid_dataset.dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  collate_fn=collator)

    encoder = Encoder(args).to(args.device)
    probe_model = TokenDetection_Classifier(encoder.hidden_size).to(args.device)

    LOGGER.info('*** Paraphrased Token Detection ***')
    LOGGER.info('Probe Type: {}'.format(str(args.probe_type)))
    LOGGER.info('Probe Layers: {}'.format(str(args.probe_layer)))
    LOGGER.info('Paremeters of Encoder: {}'.format(sum(param.numel() for param in encoder.parameters() if param.requires_grad)))
    LOGGER.info('Parameters of Probing Classifier: {}'.format(sum(param.numel() for param in probe_model.parameters() if param.requires_grad)))
    
    # Freeze Parameters of BERT
    for name, param in encoder.named_parameters():
        param.requires_grad = False

    trainable_encoder_parameters = sum(param.numel() for param in encoder.parameters() if param.requires_grad)
    trainable_mlp_parameters = sum(param.numel() for param in probe_model.parameters() if param.requires_grad)
    
    # print trainable parameters. it has to be same with MLP Parameters. 
    trainable_parameters = int(trainable_encoder_parameters) + int(trainable_mlp_parameters)
    LOGGER.info('Trainable Parameters: {}'.format(trainable_parameters))
    
    if trainable_parameters != trainable_mlp_parameters:
        LOGGER.info('The number of trainable parameters is greater than that of probing classifier parameters.')
        
    optimizer = get_adamw_optimizer(probe_model, args)
    scheduler = get_explr_scheduler(optimizer, args)
        
    # Train!
    torch.cuda.empty_cache()
    train(encoder, probe_model, train_dataloader, valid_dataloader, optimizer, scheduler, args)

if __name__ == '__main__':
    args = argument_parser()
    main(args)