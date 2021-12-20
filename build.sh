# cp -r QA_model api/
# docker-compose -d up --build api app rabbit-mq
rm -rf api/QA_model
cp -r QA_model api/
docker stop final-api
docker rm final-api
docker build --tag final-project-api ./api/
docker run --gpus all -d -p 9090:9090 --name final-api final-project-api:latest