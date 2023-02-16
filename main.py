
import dataloader
import lstm_model
import training
import inference

import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd
import spacy
import dill
import argparse
from torchtext.data.utils import get_tokenizer
torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2

def activate_inference_mode():
    
    with open("TEXT.Field","rb")as f:
        TEXT=dill.load(f)

    torch.manual_seed(RANDOM_SEED)
    model = lstm_model.RNN(input_dim=len(TEXT.vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES 
    )
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    print('[WARNING] : THE BANGLA COMMENTS MIGHT NOT BE CLEAR IN TERMINAL DUE TO ENCODING ISSUE!!!')
    with open('sarcastic_comment.txt','r+') as file:
        for sentence in file:
            not_sarcasm_probability = inference.predict_sarcasm(model,optimizer,TEXT,sentence,DEVICE)
            if not_sarcasm_probability >= 0.5:
                print(f'{sentence.strip()} is not sarcastic!')
            else:
                print(f'{sentence.strip()} is sarcastic!')


def activate_training_mode(train_loader,valid_loader,test_loader,TEXT):

    torch.manual_seed(RANDOM_SEED)
    model = lstm_model.RNN(input_dim=len(TEXT.vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES 
    )
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    training.trainer(model,optimizer,train_loader,valid_loader,test_loader,NUM_EPOCHS,DEVICE)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training',default='Y',choices=['Y','N'],type=str,help="Option to choose whether to run\
        in training mode or inference mode. Select Y for training and N for inferencing. Default is Y i.e. training mode.")

    args = parser.parse_args()
    if args.training == 'Y':
        train_loader, valid_loader, test_loader, TEXT = \
        dataloader.train_valid_test_split(RANDOM_SEED,VOCABULARY_SIZE,BATCH_SIZE,DEVICE)
        activate_training_mode(train_loader,valid_loader,test_loader,TEXT)
    else:
        activate_inference_mode()