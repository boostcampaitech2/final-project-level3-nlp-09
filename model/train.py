

import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from torch import optim
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AdamW, Trainer, TrainingArguments

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_metric
import wandb

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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed= 42
seed_everything(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df= pd.read_csv('./trans_df.csv')
print(df.tail())

df['answer']= df['answer'].replace(True, 1)
df['answer']= df['answer'].replace(False, 0)
train_df, val_df= train_test_split(df, test_size= 0.2, stratify= df['answer'], random_state= seed)

train_label= torch.tensor(list(map(int, list(train_df['answer']))))
val_label= torch.tensor(list(map(int, list(val_df['answer']))))

print(train_label.shape)
print(val_label.shape)

model_name= 'klue/roberta-large'
config= AutoConfig.from_pretrained(model_name)
tokenizer= AutoTokenizer.from_pretrained(model_name)
model= AutoModelForSequenceClassification.from_pretrained(model_name)

model.to(device)

lr= 1e-5
batch_size= 8
steps= 200
args= TrainingArguments(
    output_dir= './best_model',
    save_total_limit= 1,
    save_steps= steps,
    eval_steps= steps,
    evaluation_strategy = "steps",
    learning_rate= lr,
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size= batch_size,
    logging_dir='./logs',
    logging_steps=steps,
    num_train_epochs=10,
    load_best_model_at_end= True
)

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

accuracy_score = load_metric('accuracy')

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)

run= wandb.init(project= 'yesorno', entity= 'quarter100', name= f'please..only_qa')

trainer = Trainer(
    model,
    args,
    train_dataset=trainset,
    eval_dataset=valset,
    tokenizer=tokenizer,
    compute_metrics= compute_metrics
)

trainer.train()
run.finish()

