# Image api server

### Upload and register task to worker
```
$ /usr/bin/curl -X POST "http://127.0.0.1:8080/chat" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":\"hihi\"}"
$ curl -X POST 127.0.0.1:8080/chat -d '{"data":"안녕하세요"}'

# Response:
# { files: [ {'original': 'README.md', 'stored': 'XXXX.md'} ] }
```

### Upload and register task to worker
```
$ curl 127.0.0.1:8080/files -F "files=@./README.md"

# Response:
# { files: [ {'original': 'README.md', 'stored': 'XXXX.md'} ] }
```

### Get result of processed by worker
```
$ curl 127.0.0.1:8080/files/XXXX.md/:result

# Response: (Result file uploaded by worker)
# XXXXXX
```