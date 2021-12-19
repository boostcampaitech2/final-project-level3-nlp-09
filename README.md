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
## Convert Pytorch to ONNX(cuda env)
```sh
$ cd app
$ python convert_graph_to_onnx.py --pipeline question-answering --framework pt --model ./model/checkpoint-28500  --quantize ./model/onnx/KLRL-QA.onnx
```

## Convert ONNX to TensorRT(cuda env)
```sh
$ chmod +x trtexec_build.sh
$ ./trtexec_build.sh
```
inference를 하고 싶다면.. `inference.py` 의 argument를 확인하고 실행
(이 때 사용하고 싶은 모델을 save_model 폴더에서 찾아 argument로 입력해줘야함.)

```python
python inference.py
```




# pt file link
[구글 드라이브](https://drive.google.com/drive/folders/1zXe4xHqX7kxOZIVjb73NW0rCZ3G7uUAX?usp=sharing)

```sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ThqTAgV0NSiEhY0MzFF3XWbvvbzTdyiI" -O model.zip && rm -rf ~/cookies.txt
```
