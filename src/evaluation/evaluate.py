
import pandas as pd

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalog_coverage
)


def evaluate_recommendations(
    recommendations_by_user,
    relevant_by_user,
    catalog_ids,
    k=10
):
    """Evaluate recommendations for users with relevant test items."""

    users = [
        user_id
        for user_id in recommendations_by_user
        if relevant_by_user.get(user_id)
    ]

    if not users:
        raise ValueError(
            "No eligible users with relevant test items."
        )

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id in users:
        recommended = recommendations_by_user[user_id]
        relevant = relevant_by_user[user_id]

        precision_scores.append(
            precision_at_k(recommended, relevant, k)
        )
        recall_scores.append(
            recall_at_k(recommended, relevant, k)
        )
        ndcg_scores.append(
            ndcg_at_k(recommended, relevant, k)
        )

    return {
        f"Precision@{k}": sum(precision_scores) / len(users),
        f"Recall@{k}": sum(recall_scores) / len(users),
        f"NDCG@{k}": sum(ndcg_scores) / len(users),
        "Coverage": catalog_coverage(
            {u: recommendations_by_user[u] for u in users},
            catalog_ids
        ),
        "Evaluation_Users": len(users)
    }


def evaluate_trained_models(
    train_df,
    test_df,
    popularity_model,
    svd_model,
    movies_df,
    k=10,
    relevance_threshold=4.0
):
    """Evaluate both models on relevant test interactions."""

    import numpy as np

    from src.training.popularity_model import recommend_popular_movies
    from src.training.svd_model import recommend_svd_movies

    catalog_ids = popularity_model["movieId"].tolist()
    catalog_set = set(catalog_ids)

    # Movies already rated by each user during training
    seen_by_user = (
        train_df.groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    # Relevant test movies
    relevant_by_user = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    # Restrict evaluation to users with relevant test items
    eligible_users = sorted(relevant_by_user)

    popularity_recommendations = {}
    svd_recommendations = {}

    for user_id in eligible_users:
        seen = seen_by_user.get(user_id, set())

        popularity_recommendations[user_id] = (
            recommend_popular_movies(
                popularity_model,
                seen_movie_ids=seen,
                top_k=k
            )
        )

        svd_recommendations[user_id] = (
            recommend_svd_movies(
                svd_model,
                user_id=user_id,
                candidate_movie_ids=catalog_ids,
                seen_movie_ids=seen,
                top_k=k
            )
        )

    results = []

    for model_name, recommendations in [
        ("Popularity", popularity_recommendations),
        ("SVD", svd_recommendations)
    ]:
        metrics = evaluate_recommendations(
            recommendations,
            relevant_by_user,
            catalog_set,
            k=k
        )

        # Diversity: average pairwise Jaccard genre distance
        movie_genres = movies_df.set_index("movieId")["genres"].to_dict()

        user_diversities = []

        for movie_list in recommendations.values():
            genre_sets = []

            for movie_id in movie_list[:k]:
                genres = movie_genres.get(movie_id, "")

                if pd.isna(genres) or genres == "(no genres listed)":
                    genre_sets.append(set())
                else:
                    genre_sets.append(set(genres.split("|")))

            distances = []

            for i in range(len(genre_sets)):
                for j in range(i + 1, len(genre_sets)):
                    union = genre_sets[i] | genre_sets[j]
                    intersection = genre_sets[i] & genre_sets[j]

                    distance = (
                        1 - len(intersection) / len(union)
                        if union else 0.0
                    )
                    distances.append(distance)

            if distances:
                user_diversities.append(
                    sum(distances) / len(distances)
                )

        metrics["Diversity"] = (
            sum(user_diversities) / len(user_diversities)
            if user_diversities else 0.0
        )

        results.append({
            "Model": model_name,
            **metrics
        })

    return pd.DataFrame(results)
