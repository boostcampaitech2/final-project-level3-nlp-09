import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import StringInput, DataframeInput, JsonInput, JsonOutput

from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.frameworks.transformers import TransformersModelArtifact
import bentoml

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import sys


@env(infer_pip_packages= True)
@artifacts([PytorchModelArtifact("model"),
            PickleArtifact("tokenizer")])
class APIService(BentoService):
    # input : {{1: 'text'}}
    @bentoml.api(input=JsonInput(), output=JsonOutput())
    def predict(self, text_json):
        
        text= text_json['text']

        data= pd.read_csv('./data/inference_sample.csv')
        passage_idx= text_json['passage_idx']
        passage= data.iloc[passage_idx]['passage']
        answer= data.iloc[passage_idx]['context_name']

        if text== 'exit':
            return {'label': text, 'passage': passage, 'answer': answer}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= self.artifacts.model.to(device)
        tokenizer= self.artifacts.tokenizer
        print(device)
        prob_list= []
        result_list= []
        print(passage)

        tokenized_text= tokenizer(
        text,
        passage,
        truncation= 'only_second',
        max_length= 256,
        stride= 64,
        padding= 'max_length',
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False, 
        return_tensors= 'pt'
        )
        print(tokenized_text['input_ids'].shape)
        print(tokenized_text['attention_mask'].shape)
        
        result_list= {0: 0, 1: 0, 2: 0}
        true_cnt, false_cnt, no_cnt= 0, 0, 0

        model.eval()
        outputs= model.forward(
            input_ids= tokenized_text['input_ids'].to(device),
            attention_mask= tokenized_text['attention_mask'].to(device)
        )['logits']
        logits= outputs
        # logits= F.softmax(logits, dim= -1)
        logits= logits.detach().cpu().numpy()

        prob= logits
            
        result= np.argmax(logits, axis= -1)
        prob_list.append(prob.tolist())
        result_list[result.tolist()[0]]+=1
        print(prob_list)
        print(result_list)

        if result_list[0] == 0 and result_list[1] == 0:
            prob_label= 2

        else:
            max_prob, max_idx= 0, 0
            for k in range(len(prob_list)):
                if max_prob < max(prob_list[k][0][:2]):
                    max_prob= max(prob_list[k][0][:2])
                    max_idx= k
            prob_label= prob_list[max_idx][0].index(max_prob)

        return {'label': prob_label}                    


if __name__ == "__main__":
    bento_svc= APIService()

    model_name= 'rockmiin/ko-boolq-model'

    config= AutoConfig.from_pretrained(model_name)
    config.num_labels= 3
    model= AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer= AutoTokenizer.from_pretrained(model_name)

    bento_svc.pack('model', model)
    bento_svc.pack('tokenizer', tokenizer)

    saved_path = bento_svc.save()
    print(saved_path)    