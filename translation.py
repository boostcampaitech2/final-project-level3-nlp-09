# dataset from https://github.com/google-research-datasets/boolean-questions

from pororo import Pororo
from tqdm import tqdm
import jsonlines
import pandas as pd

# file_path : jsonl / save_path : csv
def translate(file_path, save_path):
    data= []
    with jsonlines.open(file_path) as file:
        for line in file.iter():
            data.append(line)
    print(len(data))
    df= pd.DataFrame([], columns= ['question', 'answer', 'passage'])

    # load pororo
    mt = Pororo(task="translation", lang="multi")

    # 영어 question 문장에는 물음표가 없어서 추가 / 물음표 유무에 따라 번역이 많이 다름
    q, p, a= [], [], []
    for i in tqdm(range(len(data))):
        p.append(mt(data[i]['passage'], src= 'en', tgt= 'ko'))
        q.append(mt((data[i]['question'] + '?'), src= 'en', tgt= 'ko'))
        a.append(data[i]['answer'])
        # print(data[i]['passage'])
        print((data[i]['question'] + '?'))
        print(q[-1]) 

    df['question']= q
    df['passage']= p
    df['answer']= a

    print(df)
    df.to_csv(save_path, index= False)

    return df


if __name__ =='__main__':

    train_df= translate('./data/train.jsonl', './data/trans_df_train.csv')
    dev_df= translate('./data/dev.jsonl', './data/trans_df_dev.csv')

    df= pd.concat([train_df, dev_df])
    df.to_csv('./data/trans_df_total.csv', index= False)

