
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

from airflow.sdk import DAG, task

PROJECT_DIR = Path("/content/drive/MyDrive/recommendation-pipeline")

with DAG(
    dag_id="movielens_recommendation_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dmml", "movielens", "recommendation"],
    default_args={"retries": 0},
) as dag:

    @task
    def check_project_files():
        required = [
            PROJECT_DIR / "pipeline_execution.py",
            PROJECT_DIR / "configs/config.yaml",
            PROJECT_DIR / "data/raw/ratings.csv",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(str(path))
        print("Project files verified")

    @task
    def run_recommendation_pipeline():
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "pipeline_execution.py")],
            cwd=str(PROJECT_DIR),
            check=True,
        )
        print("Recommendation pipeline completed")

    @task
    def verify_pipeline_outputs():
        required = [
            PROJECT_DIR / "pipeline_runs/automated_full_run/models/popularity_model.pkl",
            PROJECT_DIR / "pipeline_runs/automated_full_run/models/svd_model.pkl",
            PROJECT_DIR / "pipeline_runs/automated_full_run/reports/model_evaluation_results.csv",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(str(path))
        print("Model and evaluation artifacts verified")

    check_project_files() >> run_recommendation_pipeline() >> verify_pipeline_outputs()
