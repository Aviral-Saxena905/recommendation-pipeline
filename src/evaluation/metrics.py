
import math

def precision_at_k(recommended, relevant, k=10):
    """Calculate Precision@K."""
    recommended = recommended[:k]
    relevant = set(relevant)

    hits = len(set(recommended) & relevant)
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Calculate Recall@K."""
    relevant = set(relevant)

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    """Calculate NDCG@K using binary relevance."""
    relevant = set(relevant)

    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, movie_id in enumerate(recommended[:k])
        if movie_id in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommendations, catalog_ids):
    """Calculate recommendation catalog coverage."""
    catalog_ids = set(catalog_ids)

    if not catalog_ids:
        return 0.0

    recommended_items = set()

    for recommendations in all_recommendations.values():
        recommended_items.update(recommendations)

    return len(recommended_items & catalog_ids) / len(catalog_ids)
