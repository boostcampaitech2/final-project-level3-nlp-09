import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.cuda.amp import autocast

class YesOrNoModel(nn.Module):
    def __init__(self, model_name):
        super(YesOrNoModel, self).__init__()

        self.config= AutoConfig.from_pretrained(model_name)
        self.config.num_labels= 3
        self.tokenizer= AutoTokenizer.from_pretrained(model_name)
        self.backbone= AutoModelForSequenceClassification.from_pretrained(model_name, config= self.config)
    
    @autocast()
    def forward(self, input_ids, attention_mask):
        outputs= self.backbone(input_ids= input_ids, attention_mask= attention_mask)['logits']
        # outputs= F.softmax(outputs, dim= -1)

        return outputs


