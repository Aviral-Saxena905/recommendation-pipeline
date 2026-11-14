
import pandas as pd

def train_popularity_model(train_df, min_ratings=5):
    """Train a popularity-based movie recommendation model."""

    popularity = train_df.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    popularity = popularity[
        popularity["rating_count"] >= min_ratings
    ]

    popularity = popularity.sort_values(
        ["average_rating", "rating_count"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return popularity


def recommend_popular_movies(
    popularity_model, seen_movie_ids=None, top_k=10
):
    """Recommend popular movies, excluding previously seen movies."""

    if seen_movie_ids is None:
        seen_movie_ids = set()

    recommendations = popularity_model[
        ~popularity_model["movieId"].isin(seen_movie_ids)
    ]

    return recommendations.head(top_k)["movieId"].tolist()
