# Airflow

## Generate fernet key
Copy & paste to docker-compose.yaml.

```sh
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
## Build
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.0/docker-compose.yaml'
mkdir ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
docker-compose up airflow-init
docker-compose up -d
```
## Set admin user
Set admin user, if admin user doesn't exit.
```bash
#to use cli command
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.0/airflow.sh'
chmod +x airflow.sh
./airflow.sh bash

#set user config example
airflow users create \
    --username admin \
    --firstname Chungchoon \
    --lastname Hindi \
    --role Admin \
    --email relilau00@gmail.com
```

## Every files and data dir should locate in **dags** directory
```bash
#first, clone this repo
cp -r ./final-project-level3-nlp-09/* opt/airflow/dags
```

## Airflow Web Server
* localhost:8080
* default ID / PW : airflow / airflow
* Unpause **DAG:retrain** on web
