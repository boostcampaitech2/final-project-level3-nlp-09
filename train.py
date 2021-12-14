

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
import argparse

import wandb

import warnings

from dataset import get_dataset, get_loader
from model import YesOrNoModel

warnings.filterwarnings("ignore")

"""
    뭔가 코드적으로 모듈을 나눈다고 가정을 하면..
    model.py : 이건 뭐 간단하니까..
    dataset.py : get dataset으로 만들면 될 듯..?
    train.py
    inference.py
"""

def get_config():
    parser = argparse.ArgumentParser()


    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default = './save_model', 
                        help='model save dir path (default : ./save_model/)')
    parser.add_argument('--wandb_path', type= str, default= 'YesOrNoModel',
                        help='wandb graph, save_dir basic path (default: YesOrNoModel') 
    parser.add_argument('--train_path', type= str, default= './data/final_df.csv',
                        help='train csv path (default: ./data/final_df.csv')   
    parser.add_argument('--model_name', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    
    args= parser.parse_args()

    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def compute_metrics(pred, labels):
    predictions= pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)

def train(train_loader, val_loader, model, criterion, optimizer, scheduler, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(args.epochs):
        train_loss, train_acc= 0, 0
        val_loss, val_acc= 0, 0
        pbar= tqdm(enumerate(train_loader), total= len(train_loader))

        model.train()
        # for idx, data in pbar:
        #     labels= data['labels'].to(device)
        #     optimizer.zero_grad()

        #     outputs= model(
        #         input_ids= data['input_ids'].to(device),
        #         attention_mask= data['attention_mask'].to(device)
        #     )
            
        #     loss= criterion(outputs, labels)
        #     logits= outputs.detach().cpu()

        #     acc= compute_metrics(logits, labels)
        #     train_loss+= loss.item() / len(data['input_ids'])
        #     train_acc+= acc['accuracy']

        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pbar= tqdm(enumerate(val_loader), total= len(val_loader))
            for val_idx, data in val_pbar:
                labels= data['labels'].to(device)

                optimizer.zero_grad()

                outputs= model(
                    input_ids= data['input_ids'].to(device),
                    attention_mask= data['attention_mask'].to(device)
                )

                loss= criterion(outputs, labels)
                logits= outputs.detach().cpu()

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

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model, os.path.join(args.save_dir, f'model_epoch_{epoch}.pt'))



if __name__ == "__main__":

    args= get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    model= YesOrNoModel(args.model_name)
    tokenizer= AutoTokenizer.from_pretrained(args.model_name)
    model.to(device)

    criterion= torch.nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr= args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=args.lr* 0.1)

    df= pd.read_csv(args.train_path)
    trainset, valset= get_dataset(df, tokenizer, args)
    train_loader, val_loader= get_loader(trainset, valset, args)

    accuracy_score = load_metric('accuracy')

    run= wandb.init(project= 'yesorno', entity= 'quarter100', name= args.wandb_path)
    train(train_loader, val_loader, model,criterion, optimizer, scheduler, args)
    run.finish()

