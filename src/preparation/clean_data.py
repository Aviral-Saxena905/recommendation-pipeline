
import pandas as pd

def clean_dataset(dataframe):
    """Remove duplicate records and rows with missing values."""
    cleaned = dataframe.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna()
    return cleaned.reset_index(drop=True)
