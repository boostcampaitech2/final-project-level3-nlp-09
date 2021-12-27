from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer
import torch
import argparse
import os

HUGGINGFACE_AUTH_TOKEN = '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'

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_model_path', type=str, default = '/home/dain/airflow_no_docker/save_model', 
                        help='model save dir path (default : /home/dain/airflow_no_docker/save_model') 
    parser.add_argument('--model_name', type=str, default='quarter100/BoolQ_dain_test',
                        help='model type (default: quarter100/BoolQ_dain_test)')
    
    args= parser.parse_args()

    return args

if __name__ == '__main__':
    args= get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_list = os.listdir(args.load_model_path)

    nums = []
    for file in file_list :
        if "checkpoint-" in file :
            nums.append(int(file[11:]))

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    new_state_dict= torch.load(args.load_model_path+"/checkpoint-"+str(min(nums))+"/pytorch_model.bin", map_location= device)
    model.load_state_dict(new_state_dict)
    model.push_to_hub(
        "BoolQ_dain_test",
        use_temp_dir=True, 
        organization="quarter100",
        use_auth_token = HUGGINGFACE_AUTH_TOKEN
        )
        
    tokenizer= AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.push_to_hub(
        "BoolQ_dain_test",
        use_temp_dir=True, 
        organization="quarter100",
        use_auth_token = HUGGINGFACE_AUTH_TOKEN
        )