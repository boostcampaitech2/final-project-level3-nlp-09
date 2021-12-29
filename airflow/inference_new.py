from re import I
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
import argparse

#from model import YesOrNoModel

def get_config():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--load_model_path', type=str, default = '/home/dain/airflow_no_docker/save_model', 
                        help='model save dir path (default : /home/dain/airflow_no_docker/save_model)') 
    parser.add_argument('--test_path', type= str, default= '/home/dain/final-project-level3-nlp-09/data/inference_sample.csv',
                        help='train csv path (default: /home/dain/final-project-level3-nlp-09/data/inference_sample.csv')   
    parser.add_argument('--model_name', type=str, default='quarter100/BoolQ_dain_test',
                        help='model type (default: quarter100/BoolQ_dain_test)')
    
    args= parser.parse_args()

    return args


def inference(data, model, tokenizer):
    label= data['answer']
    acc= 0
    true_cnt, false_cnt, no_cnt= 0, 0, 0
    res_dict= dict()

    for i in tqdm(range(len(data))):
        if data.iloc[i]['answer'] == True:
            label= 1
        elif data.iloc[i]['answer'] == False:
            label= 0
        else: label= 2
        final_prediction= 0

        tokenized_data= tokenizer(
            data.iloc[i]['question'],
            data.iloc[i]['passage'],
            truncation= 'only_second',
            max_length= 512,
            padding= 'max_length',
            stride= 128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, 
            return_tensors= 'pt'
            )

        result_list= defaultdict(lambda: 0)
        prob_list= []
        for j in range(len(tokenized_data['input_ids'])):
            outputs= model(
                input_ids= tokenized_data['input_ids'][j].unsqueeze(0).to(device),
                attention_mask= tokenized_data['attention_mask'][j].unsqueeze(0).to(device)
            )
            logits= outputs['logits']
            logits= F.softmax(logits, dim= -1)
            logits= logits.detach().cpu().numpy()
            prob= logits
            
            result= np.argmax(logits, axis= -1)
            prob_list.append(prob.tolist())

            result_list[result.tolist()[0]]+=1
        
        if result_list[1] != 0:
            if label == 1: acc+=1; true_cnt+=1; final_prediction= 1
        elif result_list[0] != 0:
            if label == 0: acc+=1; false_cnt+=1; final_prediction= 0
        else:
            if label == 2: acc+=1; no_cnt+=1; final_prediction= 2 
        
        res_dict['YES']= result_list[1]
        res_dict['NO']= result_list[0]
        res_dict['NO ANSWER']= result_list[2]

        print(f'label: {label} prediction: {final_prediction} result list : {res_dict} prob_list {prob_list}')

    print(f'true pred: {true_cnt}/49 false pred: {false_cnt}/11 no pred: {no_cnt}/24')
    print(f'acc : {acc/len(data)}')


if __name__ == '__main__':
    args= get_config()

    data= pd.read_csv(args.test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_list = os.listdir(args.load_model_path)

    nums = []
    for file in file_list :
        if "checkpoint-" in file :
            nums.append(int(file[11:]))

    print(data.isnull().sum())
    print(data.groupby(data['answer']).count())

    #model= YesOrNoModel(args.model_name)
    #model= torch.load(args.load_model_path, map_location= device)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    new_state_dict= torch.load(args.load_model_path+"/checkpoint-"+str(min(nums))+"/pytorch_model.bin", map_location= device)
    model.load_state_dict(new_state_dict)
    tokenizer= AutoTokenizer.from_pretrained(args.model_name)

    inference(data, model, tokenizer)