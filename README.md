# twenty-questions
스무고개

# 훈련
python train.py 

# inference
python inference.py --output_dir ./outputs/one_question --model_name_or_path ./models/mrc/checkpoint-28500/ --do_predict
의 형식으로 진행
가장 밑에 메인 함수 부르는 과정에서 context와 question을 직접 선언하는 코드를 추가하였음. 한개 씩 만 해볼 수 있음.......


# pt file link
[구글 드라이브](https://drive.google.com/drive/folders/1zXe4xHqX7kxOZIVjb73NW0rCZ3G7uUAX?usp=sharing)

```sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI" -O model.zip && rm -rf ~/cookies.txt
```
