# cp -r QA_model api/
# docker-compose -d up --build api app rabbit-mq
docker stop final-project-api
docker build --tag final-project-api ./api/
docker run --gpus all -d -p 9090:9090 --name final-project-api final-project-level3-nlp-09_api:latest