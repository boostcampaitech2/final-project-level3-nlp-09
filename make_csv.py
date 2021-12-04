import pandas as pd

df_train = pd.read_excel('/opt/ml/sentiment_analysis/data/Training.xlsx') #40879개
df_test = pd.read_excel('/opt/ml/sentiment_analysis/data/Test.xlsx') #5130개

df_train.loc[35511, '감정_대분류']= '기쁨'
df_train.loc[34527, '감정_대분류']= '불안'

df_train_fillna = df_train.fillna("")
df_test_fillna = df_test.fillna("")

df_train["사람문장모음"] = df_train_fillna["사람문장1"] + " " + df_train_fillna["사람문장2"] + " " + df_train_fillna["사람문장3"] + " " + df_train_fillna["사람문장4"]
df_train["사람문장모음"] = df_train["사람문장모음"].apply(lambda x : x.rstrip())
df_test["사람문장모음"] = df_test_fillna["사람문장1"] + " " + df_test_fillna["사람문장2"] + " " + df_test_fillna["사람문장3"] + " " + df_test_fillna["사람문장4"]
df_test["사람문장모음"] = df_test["사람문장모음"].apply(lambda x : x.rstrip())

df_train.to_csv('/opt/ml/sentiment_analysis/data/train.csv', columns = ["사람문장모음", "감정_대분류", "감정_소분류"], index=False)
df_test.to_csv('/opt/ml/sentiment_analysis/data/test.csv', columns = ["사람문장모음", "감정_대분류", "감정_소분류"], index=False)