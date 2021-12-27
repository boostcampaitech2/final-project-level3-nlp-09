# hello_world.py

from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator

def print_world() -> None:
    print("world")

def xcom_pull_test(**context):
    xcom_best_return = float(context["task_instance"].xcom_pull(task_ids='inference_best_model')[6:])
    xcom_new_return = float(context["task_instance"].xcom_pull(task_ids='inference_retrained_model')[6:])

    print("xcom_best_return : {}".format(xcom_best_return))
    print("xcom_new_return : {}".format(xcom_new_return))

    return xcom_best_return < xcom_new_return

def choose_branch(**context):
    is_new_better = context["task_instance"].xcom_pull(task_ids='compare_two_models')
    branches = ['upload_retrained_model', 'nothing_happened']
    chosen = branches[0] if is_new_better else branches[1]
    print(f'chosen: {chosen}')
    return chosen

# with 구문으로 DAG 정의를 시작합니다.
with DAG(
    dag_id="retrain",  # DAG의 식별자용 아이디입니다.
    description="My third DAG",  # DAG에 대해 설명합니다.
    start_date=days_ago(1),  # DAG 정의 기준 2일 전부터 시작합니다.
    schedule_interval="0 0 * * *",  # 매일 06:00에 실행합니다.
    tags=["my_dags"],  # 태그 목록을 정의합니다. 추후에 DAG을 검색하는데 용이합니다.
) as dag:
    
    # 테스크를 정의합니다.
    # bash 커맨드로 echo hello 를 실행합니다.
    setting = BashOperator(
        task_id="setting",
        #bash_command="pip install torch",
        bash_command="sh /opt/airflow/final-project-level3-nlp-09/setting.sh ",
        owner="dain",  # 이 작업의 오너입니다. 보통 작업을 담당하는 사람 이름을 넣습니다.
        dag=dag
    )
    """
    # 테스크를 정의합니다.
    # python 함수인 print_world를 실행합니다.
    t2 = PythonOperator(
        task_id="print_world",
        python_callable=print_world,
        depends_on_past=True,
        owner="dain",
        dag=dag
    )
    t3 = BashOperator(
        task_id='print_date',
        bash_command='df -h',
        owner="dain",
        dag=dag
    )
    train_task = BashOperator(
        task_id='train_yes_or_no',
        bash_command='python /opt/airflow/final-project-level3-nlp-09/train.py',
        owner="dain",
        dag=dag
    )
    
    inference_task1 = BashOperator(
        task_id='inference_best_model',
        bash_command='python /opt/airflow/final-project-level3-nlp-09/inference_best.py',
        owner="dain",
        dag=dag
    )

    retrain_model = BashOperator(
        task_id='retrain_model',
        bash_command='echo "retrain!!"',
        #bash_command='python /opt/airflow/final-project-level3-nlp-09/train.py',
        owner="dain",
        dag=dag
    )


    inference_task2 = BashOperator(
        task_id='inference_retrained_model',
        bash_command='python /opt/airflow/final-project-level3-nlp-09/inference_new.py',
        owner="dain",
        dag=dag
    )

    compare_two_models = PythonOperator(
        task_id = 'compare_two_models',
        python_callable = xcom_pull_test,
        owner="dain",
        dag = dag
    )

    branching = BranchPythonOperator(task_id='choose_branch', python_callable=choose_branch)
    """
    upload_retrained_model = BashOperator(
        task_id='upload_retrained_model',
        bash_command='python /opt/airflow/final-project-level3-nlp-09/upload_model.py',
        owner="dain",
        dag=dag
    )

    nothing_happened = BashOperator(
        task_id='nothing_happened',
        bash_command='echo "new trained model is not better than the best model."',
        owner="dain",
        dag=dag
    )

    # 테스크 순서를 정합니다.
    # t1 실행 후 t2를 실행합니다.
    #install_requirements >> inference_task1 >> retrain_model >> inference_task2 >> compare_two_models >> branching >> [upload_retrained_model, nothing_happened]
    setting >> upload_retrained_model
    