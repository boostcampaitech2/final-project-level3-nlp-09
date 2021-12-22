# twenty-questions
# Env Setting
## Train, Inference env
```sh
pip install -r requiremnets
```

---
## Airflow

[Airflow documnet](airflow/README.md)


# API server
[API document](api/README.md)

## Demo
```sh
# Boolean QA Model
./bm-demo.sh
# Extraction-based QA Model
./em-demo.sh
```
## Build
```
$ sudo ./build.sh
$ docker-compose up -d --build rabbitmq app
```
# Model
## Train
```sh
python train.py 
```


```python
python inference.py
```




# pt file link
[구글 드라이브](https://drive.google.com/drive/folders/1zXe4xHqX7kxOZIVjb73NW0rCZ3G7uUAX?usp=sharing)

```sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI" -O model.zip && rm -rf ~/cookies.txt
```
