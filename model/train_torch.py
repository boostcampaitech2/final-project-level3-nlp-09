

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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_metric
from tqdm import tqdm
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


df= pd.read_csv('./trans_df_9000.csv')
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
epochs= 5

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

# def compute_metrics(pred):
#     predictions, labels = pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy_score.compute(predictions=predictions, references=labels)
def compute_metrics(pred, labels):
    predictions= pred
    predictions = np.argmax(predictions, axis=1)
    # print(predictions)
    return accuracy_score.compute(predictions=predictions, references=labels)

train_loader= DataLoader(trainset, batch_size= batch_size, shuffle= True, pin_memory= True)
val_loader= DataLoader(valset, batch_size= batch_size, shuffle= True, pin_memory= True)


criterion= torch.nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr= lr)


run= wandb.init(project= 'yesorno', entity= 'quarter100', name= f'9000_torch_ver')

for epoch in range(epochs):
    train_loss, train_acc= 0, 0
    val_loss, val_acc= 0, 0
    pbar= tqdm(enumerate(train_loader), total= len(train_loader))

    model.train()
    for idx, data in pbar:
        labels= data['labels'].cpu()

        optimizer.zero_grad()

        outputs= model(
            input_ids= data['input_ids'].to(device),
            attention_mask= data['attention_mask'].to(device)
        )

        loss= criterion(outputs['logits'], labels.to(device))
        logits= outputs['logits'].detach().cpu()
        acc= compute_metrics(logits, labels)
        train_loss+= loss.item() / len(data['input_ids'])
        
        # print(acc)
        # print(labels)
        train_acc+= acc['accuracy']

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pbar= tqdm(enumerate(val_loader), total= len(val_loader))
        for val_idx, data in val_pbar:
            labels= data['labels'].cpu()

            optimizer.zero_grad()

            outputs= model(
                input_ids= data['input_ids'].to(device),
                attention_mask= data['attention_mask'].to(device)
            )

            logits= outputs['logits'].detach().cpu()
            loss= criterion(logits, labels)

            acc= compute_metrics(logits, labels)
            val_loss+= loss.item() / len(data['input_ids'])
            val_acc+= acc['accuracy']
            
    train_loss_= train_loss / len(train_loader)
    train_acc_= train_acc / len(train_loader)
    val_loss_= val_loss / len(val_loader)
    val_acc_= val_acc / len(val_loader)

    print(f'epoch: {epoch}, train_loss: {train_loss_}, train_acc: {train_acc_}, \
        val_loss: {val_loss_}, val_acc: {val_acc_}')
    
    wandb.log({'train/accuracy': train_acc_, 'train/loss': train_loss_, 'eval/loss': val_loss_,
            'eval/accuracy': val_acc_})
    torch.save(model, f'./model_epoch_{epoch}.pt')



# run.finish()

