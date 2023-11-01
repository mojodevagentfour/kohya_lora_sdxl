from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError
from Dagster_pipline import generateuploader_form
import time

app = Celery('dagster_tasks', broker='redis://localhost:6379/0')


@app.task(bind=True)  # Bind to access self, set max_retries
def execute_pipeline(self, data):
    try:
        result = generateuploader_form.execute_in_process(run_config={"resources": {"order_data": {"config": data}}})
    except ValueError as e:
        print(f"Failed to execute pipline Reason:{e}")
