
import os
import pandas as pd

def generate_user_features(train_df):
    """Generate user features from training data."""

    return train_df.groupby("userId").agg(
        user_avg_rating=("rating", "mean"),
        user_activity_count=("rating", "count")
    ).reset_index()


def generate_item_features(train_df):
    """Generate movie features from training data."""

    return train_df.groupby("movieId").agg(
        item_avg_rating=("rating", "mean"),
        item_popularity=("rating", "count"),
        unique_user_count=("userId", "nunique")
    ).reset_index()


def generate_interaction_features(train_df):
    """Generate interaction-level features."""

    columns = [
        "userId", "movieId", "rating",
        "timestamp", "year", "month", "day"
    ]

    return train_df[columns].copy()


def save_features(features, project_dir):
    """Save feature datasets in CSV and Parquet formats."""

    feature_dir = os.path.join(project_dir, "data", "features")
    os.makedirs(feature_dir, exist_ok=True)

    for name, dataframe in features.items():
        dataframe.to_csv(
            os.path.join(feature_dir, f"{name}.csv"),
            index=False
        )

        dataframe.to_parquet(
            os.path.join(feature_dir, f"{name}.parquet"),
            index=False
        )

    return True
