FROM python:3.8

ADD ./ /app

# RUN apt-get update -y && \
#     apt-get install -y python-pip python-dev
RUN pip3 install -r /app/requirements.txt --no-cache-dir

WORKDIR /app
# ENTRYPOINT [ "python" ]
CMD ["python", "myapp.py" ]

EXPOSE 5000

# docker build --tag test ./api -> test docker image
# docker run test -> container