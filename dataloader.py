import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import spacy
import torchtext
import random
import dill

def save_test_data(test_data):
    '''
    Saving test dataset into 2 files, all sarcastic comments in sarcastic_comment.txt
    and all non-sarcastic comments in non_sarcastic_comment.txt. During inference, these 
    files will be read.
    '''
    for i, x in enumerate(test_data):
        if x.LABEL_COLUMN_NAME == '1':
            # print(x.TEXT_COLUMN_NAME)
            comment = ""
            for word in x.TEXT_COLUMN_NAME:
                comment+=word
            with open('sarcastic_comment.txt','a') as f:
                f.write(comment)
                f.write('\n')
        else:
            comment=""
            for word in x.TEXT_COLUMN_NAME:
                comment+=word
            with open('not_sarcastic_comment.txt','a') as t:
                t.write(comment)
                t.write('\n')

def train_valid_test_split(RANDOM_SEED,VOCABULARY_SIZE,BATCH_SIZE,DEVICE):

    data = pd.read_csv('sarcasm_detection.csv')
    data.rename(columns={'Id':'ID','headlines':'TEXT_COLUMN_NAME','category':'LABEL_COLUMN_NAME'},inplace=True)
    data.drop(axis=1,columns=['ID'], inplace=True)
    data.to_csv('dataset.csv',index=False)

    TEXT = torchtext.legacy.data.Field(
        tokenize='spacy',
        tokenizer_language='xx_ent_wiki_sm' #tokenizer for Bangla text
    )
    LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)
    fields = [('TEXT_COLUMN_NAME',TEXT),('LABEL_COLUMN_NAME',LABEL)]
    dataset = torchtext.legacy.data.TabularDataset(
        path='dataset.csv',
        format='csv',
        skip_header=True,
        fields=fields
    )

    train_data,test_data = dataset.split(
        split_ratio=[0.8,0.2],
        random_state=random.seed(RANDOM_SEED)
    )
    train_data, valid_data = train_data.split(
    split_ratio=[0.85, 0.15],
    random_state=random.seed(RANDOM_SEED))

    save_test_data(test_data)
    
    TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
    LABEL.build_vocab(train_data)

    train_loader, valid_loader, test_loader = \
    torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
         batch_size=BATCH_SIZE,
         sort_within_batch=False,
         sort_key=lambda x: len(x.TEXT_COLUMN_NAME),
         device=DEVICE
    )

    with open('TEXT.Field',"wb") as f:
        dill.dump(TEXT,f)
    
    return train_loader, valid_loader, test_loader, TEXT