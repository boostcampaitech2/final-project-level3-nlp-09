cp -r QA_model api/
docker-compose -d up --build api app rabbit-mq
