
import os
import mlflow


def log_pipeline_execution(project_dir, pipeline_results, dry_run=True):
    """Log completed pipeline execution metrics to MLflow."""

    database_path = os.path.join(project_dir, "mlflow.db")

    mlflow.set_tracking_uri(
        "sqlite:///" + database_path
    )

    experiment_name = "MovieLens_Pipeline_Execution"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Pipeline_Dry_Run" if dry_run else "Pipeline_Full_Run") as run:

        mlflow.log_param("execution_mode", "dry_run" if dry_run else "full_run")
        mlflow.log_param("dataset", "MovieLens Latest Small")

        for metric_name, metric_value in pipeline_results.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(
                    metric_name,
                    float(metric_value)
                )

        return run.info.run_id
