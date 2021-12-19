# Extractived-based MRC Api Server
## Demo
```sh
uvicorn main:app --host 0.0.0.0 --port 9090
```
## API
```sh
# /get_category
# Request by GET
# Requst: None
# Return: {
#     'category': str, 
#     'context': str, 
#     'answer': str
# }
0.0.0.0:9090/get_category

# /chat
# Request json by POST
# Request: {"data": "Any quesiton you want"}
# Return: {
#     'result': str
# }
0.0.0.0:9090/chat
```
### Upload and register task to worker
```sh
# Set Context
$ curl "http://14.49.45.218:9090/set_category"

# Set Question
# 정답 마스킹 전에 모델이 제대로 돌아가는지 확인할 수 있는 가장 좋은 질문
$ curl -X POST "http://14.49.45.218:9090/chat" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":\"이것의 이름은 무엇인가요?\"}"
```