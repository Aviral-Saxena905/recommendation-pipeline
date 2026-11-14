
import pandas as pd

def time_based_split(ratings_df, test_size=0.2):
    """Split ratings chronologically into training and testing datasets."""

    sorted_df = ratings_df.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    split_index = int(len(sorted_df) * (1 - test_size))

    train_df = sorted_df.iloc[:split_index].copy()
    test_df = sorted_df.iloc[split_index:].copy()

    return train_df, test_df
