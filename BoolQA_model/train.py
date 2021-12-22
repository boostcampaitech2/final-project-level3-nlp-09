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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import wandb

import warnings

from dataset import get_dataset

warnings.filterwarnings("ignore")

def get_config():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default = './save_model', 
                        help='model save dir path (default : ./save_model/)')
    parser.add_argument('--wandb_path', type= str, default= 'YesOrNoModel_hg',
                        help='wandb graph, save_dir basic path (default: YesOrNoModel') 
    parser.add_argument('--train_path', type= str, default= './data/final_df.csv',
                        help='train csv path (default: ./data/final_df.csv')   
    parser.add_argument('--model_name', type=str, default='rockmiin/ko-boolq-model',
                        help='model type (default: klue/roberta-large)')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--gradient_accum', type=int, default=2,
                        help='gradient accumulation (default: 2)')
    parser.add_argument('--batch_valid', type=int, default=32,
                        help='input batch size for validing (default: 32)')
    parser.add_argument('--eval_steps', type=int, default=300,
                        help='eval_steps (default: 250)')
    parser.add_argument('--save_steps', type=int, default=300,
                        help='save_steps (default: 250)')
    parser.add_argument('--logging_steps', type=int,
                        default=100, help='logging_steps (default: 50)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='weight_decay (default: 0.01)')
    parser.add_argument('--metric_for_best_model', type=str, default='accuracy',
                        help='metric_for_best_model (default: accuracy')
    
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

def compute_metrics(pred):
    """ validation을 위한 metrics function """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    acc = accuracy_score(labels, preds) 

    return {
        'accuracy': acc,
    }

if __name__ == "__main__":

    args= get_config()
    save_dir= f'./save_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    config= AutoConfig.from_pretrained(args.model_name)
    config.num_labels= 3
    model= AutoModelForSequenceClassification.from_pretrained(args.model_name, config= config)
    tokenizer= AutoTokenizer.from_pretrained(args.model_name)
    model.to(device)

    df= pd.read_csv(args.train_path)
    trainset, valset= get_dataset(df, tokenizer, args)

    training_args= TrainingArguments(
            output_dir= save_dir,
            save_total_limit= 1,
            gradient_accumulation_steps= args.gradient_accum,
            save_steps=args.save_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_valid,
            label_smoothing_factor=0.1,
            logging_dir='./logs',
            logging_steps=args.logging_steps,
            metric_for_best_model= args.metric_for_best_model,
            evaluation_strategy= 'steps',
            eval_steps= args.eval_steps,
            load_best_model_at_end=True
        )

    trainer= Trainer(
                model= model,
                args= training_args,
                train_dataset= trainset,
                eval_dataset= valset,
                compute_metrics= compute_metrics
            )
    trainer.save_model()  # Saves the tokenizer too for easy upload

    run= wandb.init(project= 'yesorno', entity= 'quarter100', name= args.wandb_path)
    trainer.train()
    run.finish()

