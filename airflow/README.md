# Airflow

## Build (Docker)
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.2.0/docker-compose.yaml'
mkdir ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
docker-compose up airflow-init
docker-compose up -d

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

## Build (Virtual Env)
```bash
source .venv/bin/activate
pip install pip --upgrade
pip install 'apache-airflow==2.2.0'
export AIRFLOW_HOME=.
airflow db init

#set user config example
airflow users create \
    --username admin \
    --firstname Chungchoon \
    --lastname Hindi \
    --role Admin \
    --email relilau00@gmail.com

#run webserver & scheduler
airflow webserver --port 8080
airflow scheduler
```

## retrain.py should locate in **dags** directory
```bash
#first, clone this repo
cp -r ./final-project-level3-nlp-09/retrain.py opt/airflow/dags/retrain.py
```

## Airflow Web Server
* localhost:8080
* default ID / PW : airflow / airflow
* Unpause **DAG:retrain** on web
