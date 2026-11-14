# Saved Recommendation Models

This directory is reserved for trained recommendation model artifacts.

The project implements two recommendation approaches:

- **Popularity-based recommender:** A baseline model using historical movie popularity.
- **SVD recommender:** A personalized collaborative filtering model implemented using Surprise.

## Generating the Models

After preparing the MovieLens input datasets and installing the project dependencies, run:

```bash
python pipeline_execution.py --full-run
```

The full pipeline trains and saves the recommendation models.

The automated full-run artifacts are stored under:

- `pipeline_runs/automated_full_run/models/popularity_model.pkl`
- `pipeline_runs/automated_full_run/models/svd_model.pkl`

Model evaluation results are available in `reports/model_evaluation_results.csv`.
