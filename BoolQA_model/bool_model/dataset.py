import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from torch import optim
from torch.cuda import device_count
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AdamW, Trainer, TrainingArguments

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_metric
from tqdm import tqdm

class Dataset:
    def __init__(self, data, labels): # data : dict, label : list느낌..
        self.data= data
        self.labels= labels
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item
    
    def __len__(self):
        return len(self.labels)

def get_dataset(df: pd.DataFrame, tokenizer, args):
    train_df, val_df= train_test_split(df, test_size= 0.2, stratify= df['answer'], random_state= args.seed)

    train_label= torch.tensor(list(map(int, list(train_df['answer']))))
    val_label= torch.tensor(list(map(int, list(val_df['answer']))))

    tokenized_train= tokenizer(
    list(train_df['question']),
    list(train_df['passage']),
    truncation= 'only_second',
    max_length= 512,
    padding= 'max_length',
    return_tensors= 'pt'
    )

    tokenized_val= tokenizer(
        list(val_df['question']),
        list(val_df['passage']),
        truncation= 'only_second',
        max_length= 512,
        padding= 'max_length',
        return_tensors= 'pt'
        )

    trainset= Dataset(tokenized_train, train_label)
    valset= Dataset(tokenized_val, val_label)

    return trainset, valset

def get_loader(trainset, valset, args):
    train_loader= DataLoader(trainset, batch_size= args.batch_size, shuffle= True)
    val_loader= DataLoader(valset, batch_size= args.batch_size, shuffle= True)

    return train_loader, val_loader