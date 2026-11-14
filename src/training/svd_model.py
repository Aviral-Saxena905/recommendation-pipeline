
from surprise import Dataset, Reader, SVD

def train_svd_model(
    train_df,
    n_factors=50,
    n_epochs=20,
    random_state=42,
    lr_all=0.005,
    reg_all=0.02
):
    """Train a personalized SVD recommendation model."""

    reader = Reader(rating_scale=(0.5, 5.0))

    dataset = Dataset.load_from_df(
        train_df[["userId", "movieId", "rating"]],
        reader
    )

    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state
    )

    model.fit(trainset)

    return model


def recommend_svd_movies(
    model, user_id, candidate_movie_ids,
    seen_movie_ids=None, top_k=10
):
    """Generate personalized movie recommendations."""

    if seen_movie_ids is None:
        seen_movie_ids = set()

    predictions = []

    for movie_id in candidate_movie_ids:
        if movie_id not in seen_movie_ids:
            score = model.predict(
                user_id, movie_id
            ).est

            predictions.append((movie_id, score))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return [
        movie_id
        for movie_id, score in predictions[:top_k]
    ]
