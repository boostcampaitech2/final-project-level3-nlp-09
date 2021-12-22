# API Server
## Boolean QA Api Server
### Demo
```sh
cd
uvicorn main:app --host 0.0.0.0 --port 9091
```
### API
```sh
# /get_category
# Request by GET
# Requst: None
# Return: {
#     'category': str, 
#     'context_name': str, 
# }
0.0.0.0:9091/set_category_boolq

# /chat
# Request json by POST
# Request: {"data": "Any quesiton you want"}
# Return: {
#     'result': str
# }
0.0.0.0:9091/chat
```
### Testing
```sh
# Set Context
$ curl -X POST "http://0.0.0.0:9091/set_category_boolq" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"category\":\"행성\", \"context_name\":\"금성\"}"

# Set Question
$ curl -X POST "http://0.0.0.0:9091/chat_boolq" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":\"이것은 행성인가요?\"}"
```

## Extractived-based MRC Api Server
### Demo
```sh
uvicorn main:app --host 0.0.0.0 --port 9090
```
### API
```sh
# /get_category
# Request by GET
# Requst: None
# Return: {
#     'category': str, 
#     'context_name': str, 
# }
0.0.0.0:9090/set_category

# /chat
# Request json by POST
# Request: {"data": "Any quesiton you want"}
# Return: {
#     'result': str
# }
0.0.0.0:9090/chat
```
### Testing
```sh
# Set Context
$ curl "http://0.0.0.0:9090/set_category" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"category\":\"mbti\", \"context_name\":\"ESFJ\"}"

# Set Question
# 정답 마스킹 전에 모델이 제대로 돌아가는지 확인할 수 있는 가장 좋은 질문
$ curl -X POST "http://0.0.0.0:9090/chat" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":\"이것은 어떤 성격인가요??\"}"
```



