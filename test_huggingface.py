from datasets import load_dataset
from datasets import Dataset

# api key 가져오기
f = open('./hf_key.txt','r')
api_key = f.readline()
f.close()

# 학습이 끝난 후 log data를 지우기 위해 실행시키는 함수
def reset_data():
    init_data = {}
    d = Dataset.from_dict(init_data)
    d.push_to_hub("quarter100/boolq_log",token = api_key)

# 추가해야하는 데이터
new_log = {'answer_keyword':['태권도','태권도'],'question':['td','t'],'answers':[0,1],'feedback':[1,1]}
new_log_data = Dataset.from_dict(new_log)

try: # 이어쓰기
    # 최신버전 가져오기
    dataset = load_dataset("quarter100/boolq_log")
    # log 추가
    for new_data in new_log_data:
        dataset['train'] = dataset['train'].add_item(new_data)
    # 새로운 data 업로드
    dataset.push_to_hub("quarter100/boolq_log",token = api_key)

except ValueError: # 재학습 시켜서 log가 비어있는 경우
    new_log_data.push_to_hub("quarter100/boolq_log",token = api_key)
