
import joblib
import pandas as pd
from src.evaluation.evaluate import evaluate_trained_models

from src.training.training_pipeline import execute_model_training
from src.evaluation.mlflow_tracking import log_pipeline_execution

import os
import sys
import yaml
import logging

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.ingestion.data_loader import load_raw_data
from src.preparation.clean_data import clean_dataset
from src.preparation.split_data import time_based_split
from src.transformation.transform_data import transform_ratings
from src.features.feature_store import (
    generate_user_features,
    generate_item_features,
    generate_interaction_features
)

from src.validation.schema import (
    ratings_schema,
    movies_schema,
    users_schema,
    tags_schema,
    links_schema
)

from src.evaluation.model_integration import evaluate_saved_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_pipeline(dry_run=True):

    config_path = os.path.join(
        PROJECT_DIR, "configs", "config.yaml"
    )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    logger.info("Pipeline started")
    logger.info("Dry run: %s", dry_run)

    # Stage 1: Ingestion
    datasets = load_raw_data(PROJECT_DIR)

    logger.info("Data ingestion completed")

    # Stage 2: Validation
    schemas = {
        "ratings": ratings_schema,
        "movies": movies_schema,
        "users": users_schema,
        "tags": tags_schema,
        "links": links_schema
    }

    for name, schema in schemas.items():
        schema.validate(datasets[name], lazy=True)
        logger.info("Validation passed: %s", name)

    # Stage 3: Cleaning
    ratings = clean_dataset(datasets["ratings"])
    movies = clean_dataset(datasets["movies"])
    users = clean_dataset(datasets["users"])

    logger.info("Data cleaning completed")

    # Stage 4: Time-based split
    test_size = config["preparation"]["test_size"]

    train_df, test_df = time_based_split(
        ratings,
        test_size=test_size
    )

    if train_df["timestamp"].max() > test_df["timestamp"].min():
        raise ValueError("Time leakage detected")

    logger.info(
        "Split completed: train=%s, test=%s",
        len(train_df),
        len(test_df)
    )

    # Stage 5: Transformation
    train_transformed = transform_ratings(
        train_df, movies, users
    )

    test_transformed = transform_ratings(
        test_df, movies, users
    )

    logger.info("Data transformation completed")

    # Stage 6: Feature generation
    user_features = generate_user_features(train_transformed)
    item_features = generate_item_features(train_transformed)
    interaction_features = generate_interaction_features(
        train_transformed
    )

    logger.info(
        "Feature generation completed: users=%s, items=%s, interactions=%s",
        len(user_features),
        len(item_features),
        len(interaction_features)
    )


    # Stage 7: Model execution
    if dry_run:
        model_results = evaluate_saved_models(PROJECT_DIR)

        logger.info(
            "Saved model evaluation completed: RMSE=%.4f, MAE=%.4f",
            model_results["svd_rmse"],
            model_results["svd_mae"]
        )

    else:
        output_dir = os.path.join(
            PROJECT_DIR, "pipeline_runs", "automated_full_run"
        )

        # Save processed datasets and generated features
        data_dir = os.path.join(output_dir, "data", "processed")
        features_dir = os.path.join(output_dir, "data", "features")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)

        train_df.to_parquet(
            os.path.join(data_dir, "train_data.parquet"),
            index=False
        )
        test_df.to_parquet(
            os.path.join(data_dir, "test_data.parquet"),
            index=False
        )

        for name, df in {
            "user_features": user_features,
            "item_features": item_features,
            "interaction_features": interaction_features
        }.items():
            df.to_parquet(
                os.path.join(features_dir, f"{name}.parquet"),
                index=False
            )

        logger.info("Fresh processed data and features saved")
        logger.info("Starting full model training")

        model_results = execute_model_training(
            train_df,
            test_df,
            output_dir,
            config
        )

        logger.info(
            "Fresh model training completed: RMSE=%.4f, MAE=%.4f",
            model_results["svd_rmse"],
            model_results["svd_mae"]
        )

        logger.info("New models saved in: %s", output_dir)

        # Load freshly trained models
        popularity_model = joblib.load(
            os.path.join(output_dir, "models", "popularity_model.pkl")
        )

        svd_model = joblib.load(
            os.path.join(output_dir, "models", "svd_model.pkl")
        )

        # Evaluate newly trained models
        evaluation_df = evaluate_trained_models(
            train_df,
            test_df,
            popularity_model,
            svd_model,
            movies,
            k=config["evaluation"]["k"]
        )

        # Include rating prediction metrics
        evaluation_df["RMSE"] = float("nan")
        evaluation_df["MAE"] = float("nan")

        svd_mask = evaluation_df["Model"] == "SVD"

        evaluation_df.loc[
            svd_mask, "RMSE"
        ] = model_results["svd_rmse"]

        evaluation_df.loc[
            svd_mask, "MAE"
        ] = model_results["svd_mae"]

        # Save fresh evaluation report
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(
            reports_dir, "model_evaluation_results.csv"
        )

        evaluation_df.to_csv(report_path, index=False)

        logger.info("Fresh evaluation report saved: %s", report_path)

        # Include fresh ranking metrics in pipeline results
        for _, row in evaluation_df.iterrows():
            prefix = row["Model"].lower()

            for metric in [
                "Precision@10",
                "Recall@10",
                "NDCG@10",
                "Coverage",
                "Diversity"
            ]:
                model_results[
                    f"{prefix}_{metric.replace('@', '_at_')}"
                ] = float(row[metric])

    if dry_run:
        logger.info(
            "DRY RUN SUCCESSFUL - No files were overwritten"
        )

    result = {
        "train_rows": len(train_transformed),
        "test_rows": len(test_transformed),
        "user_features": len(user_features),
        "item_features": len(item_features),
        "interaction_features": len(interaction_features),
        "svd_rmse": round(model_results["svd_rmse"], 4),
        "svd_mae": round(model_results["svd_mae"], 4)
    }

    if not dry_run:
        for key, value in model_results.items():
            if key.startswith(("popularity_", "svd_")):
                if isinstance(value, (int, float)):
                    result[key] = value

    return result


if __name__ == "__main__":
    dry_run = "--full-run" not in sys.argv
    result = run_pipeline(dry_run=dry_run)

    print("\nPIPELINE SUMMARY")
    for key, value in result.items():
        print(f"{key}: {value}")

    run_id = log_pipeline_execution(PROJECT_DIR, result, dry_run=dry_run)

    print("\nMLflow tracking completed!")
    print("MLflow Run ID:", run_id)
