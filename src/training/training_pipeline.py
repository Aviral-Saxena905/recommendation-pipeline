
import os
import joblib

from surprise import accuracy

from src.training.popularity_model import train_popularity_model
from src.training.svd_model import train_svd_model


def execute_model_training(
    train_df,
    test_df,
    output_dir,
    config
):
    """Train and evaluate recommendation models."""

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Read model parameters from configuration
    popularity_config = config["models"]["popularity"]
    svd_config = config["models"]["svd"]

    # Train popularity model
    popularity_model = train_popularity_model(
        train_df,
        min_ratings=popularity_config["min_ratings"]
    )

    # Train SVD model
    svd_model = train_svd_model(
        train_df,
        n_factors=svd_config["n_factors"],
        n_epochs=svd_config["n_epochs"],
        random_state=svd_config["random_state"],
        lr_all=svd_config["lr_all"],
        reg_all=svd_config["reg_all"]
    )

    # Evaluate SVD on the testing dataset
    testset = list(
        test_df[["userId", "movieId", "rating"]]
        .itertuples(index=False, name=None)
    )

    predictions = svd_model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    # Save models in the selected output directory
    joblib.dump(
        popularity_model,
        os.path.join(models_dir, "popularity_model.pkl")
    )

    joblib.dump(
        svd_model,
        os.path.join(models_dir, "svd_model.pkl")
    )

    return {
        "popularity_movies": len(popularity_model),
        "svd_rmse": rmse,
        "svd_mae": mae,
        "training_rows": len(train_df),
        "testing_rows": len(test_df)
    }
