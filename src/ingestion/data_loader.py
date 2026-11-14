
import os
import pandas as pd

def load_raw_data(project_dir):
    """Load MovieLens datasets from CSV files."""

    raw_dir = os.path.join(project_dir, "data", "raw")

    datasets = {}

    for name in ["movies", "ratings", "tags", "links", "users"]:
        file_path = os.path.join(raw_dir, f"{name}.csv")
        datasets[name] = pd.read_csv(file_path)

    return datasets


def save_processed_data(datasets, project_dir):
    """Save datasets in Parquet format."""

    processed_dir = os.path.join(project_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for name, dataframe in datasets.items():
        output_path = os.path.join(
            processed_dir, f"{name}.parquet"
        )
        dataframe.to_parquet(output_path, index=False)

    return True
