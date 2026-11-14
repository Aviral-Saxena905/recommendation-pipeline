
import pandas as pd

from schema import (
    ratings_schema,
    users_schema,
    movies_schema,
    tags_schema,
    links_schema
)


def validate_dataset(dataframe, schema, dataset_name):
    try:
        validated_data = schema.validate(dataframe)

        print(f"{dataset_name}: VALIDATION PASSED")

        return validated_data

    except Exception as error:
        print(f"{dataset_name}: VALIDATION FAILED")
        print(error)

        return None
