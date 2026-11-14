
import pandas as pd

def transform_ratings(ratings_df, movies_df, users_df):
    """Add date features and movie/user information."""

    transformed = ratings_df.copy()

    transformed["datetime"] = pd.to_datetime(
        transformed["timestamp"], unit="s"
    )

    transformed["year"] = transformed["datetime"].dt.year
    transformed["month"] = transformed["datetime"].dt.month
    transformed["day"] = transformed["datetime"].dt.day

    transformed = transformed.merge(
        movies_df, on="movieId", how="left"
    )

    transformed = transformed.merge(
        users_df, on="userId", how="left"
    )

    return transformed
