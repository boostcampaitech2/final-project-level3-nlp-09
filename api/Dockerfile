FROM python:3.7

ADD ./ /api

RUN pip3 install -r /api/requirements.txt --no-cache-dir

WORKDIR /api
CMD uvicorn main:app --host 0.0.0.0 --port 8080

EXPOSE 8080