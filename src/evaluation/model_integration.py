
import os
import joblib
import pandas as pd
from surprise import accuracy


def evaluate_saved_models(project_dir):
    """Load saved models and verify their evaluation results."""

    models_dir = os.path.join(project_dir, "models")

    popularity_model = joblib.load(
        os.path.join(models_dir, "popularity_model.pkl")
    )

    svd_model = joblib.load(
        os.path.join(models_dir, "svd_model.pkl")
    )

    test_df = pd.read_parquet(
        os.path.join(
            project_dir, "data", "processed", "test_data.parquet"
        )
    )

    testset = list(
        test_df[["userId", "movieId", "rating"]]
        .itertuples(index=False, name=None)
    )

    predictions = svd_model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    evaluation_report = pd.read_csv(
        os.path.join(
            project_dir, "reports", "model_evaluation_results.csv"
        )
    )

    if not {"Popularity", "SVD"}.issubset(
        set(evaluation_report["Model"])
    ):
        raise ValueError("Missing model evaluation results")

    return {
        "popularity_movies": len(popularity_model),
        "svd_rmse": rmse,
        "svd_mae": mae,
        "evaluated_test_rows": len(test_df),
        "evaluation_report": evaluation_report
    }
