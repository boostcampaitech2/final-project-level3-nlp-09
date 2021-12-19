# Image api server

## Overview

- main.py
    - inference model
- inference.py
    - i
### Upload and register task to worker
```
$ /usr/bin/curl -X POST "http://0.0.0.0:9090/chat" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":\"hihi\"}"
$ /usr/bin/curl -X POST "http://172.28.181.196:8080/chat" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\":\"hihi\"}"

# Response:
# { files: [ {'original': 'README.md', 'stored': 'XXXX.md'} ] }
```