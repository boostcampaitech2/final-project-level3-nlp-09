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
import json

from model import YesOrNoModel

def get_config():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--load_model_path', type=str, default = './save_model/model_epoch_5.pt', 
                        help='model save dir path (default : ./save_model/model_epoch_3.pt)') 
    parser.add_argument('--test_path', type= str, default= './data/inference_sample.csv',
                        help='train csv path (default: ./data/inference_sample.csv')   
    parser.add_argument('--model_name', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    
    args= parser.parse_args()

    return args


def inference(data, model, tokenizer):
    label= data['answer']
    acc= 0
    return_dict= dict()


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

        result_list= {0: 0, 1: 0, 2: 0}
        prob_list= []
        decoded_feature= {
            0: {'text': [], 'text_idx': [],'predict': []},
            1: {'text': [], 'text_idx': [],'predict': []},
            2: {'text': [], 'text_idx': [],'predict': []}
            }
        for j in range(len(tokenized_data['input_ids'])):
            outputs= model(
                input_ids= tokenized_data['input_ids'][j].unsqueeze(0).to(device),
                attention_mask= tokenized_data['attention_mask'][j].unsqueeze(0).to(device)
            )
            logits= outputs
            logits= F.softmax(logits, dim= -1)
            logits= logits.detach().cpu().numpy()
            prob= logits
            
            result= np.argmax(logits, axis= -1)
            prob_list.append(prob.tolist())

            result_list[result.tolist()[0]]+=1

            decoded_data= tokenizer.decode(tokenized_data['input_ids'][j], skip_special_tokens= True)
            for h in range(3):
                if result.tolist()[0] == h:
                    decoded_feature[h]['text'].append(decoded_data)
                    decoded_feature[h]['text_idx'].append(j)
                    decoded_feature[h]['predict'].append(result.tolist()[0])
        # print(decoded_feature)
        
        # if result_list[0] == 0 and result_list[1] == 0:
        #     prob_label= 2
        # else:
        #     max_prob, max_idx= 0, 0
        #     for k in range(len(prob_list)):
        #         if max_prob < max(prob_list[k][0][:2]):
        #             max_prob= max(prob_list[k][0][:2])
        #             max_idx= k
        #     prob_label= prob_list[max_idx][0].index(max_prob)

        # if prob_label == label: 
        #     if label==0: acc+=1; false_cnt+= 1
        #     elif label==1: acc+=1; true_cnt+= 1
        #     elif label==2: acc+=1; no_cnt +=1

        if result_list[1] != 0:
            if label == 1: 
                acc+=1; true_cnt+=1; final_prediction= 1
                return_dict[i]= decoded_feature[label].copy()
                return_dict[i]['answer']= label

        elif result_list[0] != 0:
            if label == 0: 
                acc+=1; false_cnt+=1; final_prediction= 0
                return_dict[i]= decoded_feature[label].copy()
                return_dict[i]['answer']= label
                
        else:
            if label == 2: 
                acc+=1; no_cnt+=1; final_prediction= 2 
                return_dict[i]= decoded_feature[label].copy()
                return_dict[i]['answer']= label

        res_dict['YES']= result_list[1]
        res_dict['NO']= result_list[0]
        res_dict['NO ANSWER']= result_list[2]

        # print(f'label: {label} prediction: {prob_label} max prob: {max_prob} result list : {res_dict}')

        print(f'label: {label} prediction: {final_prediction} result list : {res_dict}')

    print(f'acc : {acc/len(data)}')
    print(f'true pred: {true_cnt}/49 false pred: {false_cnt}/11 no pred: {no_cnt}/24')

    if not os.path.exists('./return_json'):
        os.makedirs('./return_json')

    with open('./return_json/return_dict.json', 'w') as f:
        f.write(json.dumps(return_dict, ensure_ascii= False, indent=4, sort_keys=True))
        # json.dump(return_dict, f, ensure_ascii=False, indent=4, sort_keys=True)

if __name__ == '__main__':
    args= get_config()

    data= pd.read_csv(args.test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(data.isnull().sum())
    print(data.groupby(data['answer']).count())

    model= YesOrNoModel(args.model_name)
    model= torch.load(args.load_model_path, map_location= device)
    tokenizer= AutoTokenizer.from_pretrained(args.model_name)

    inference(data, model, tokenizer)