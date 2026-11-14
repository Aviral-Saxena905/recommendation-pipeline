
import os
import sys
import yaml

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, PROJECT_DIR)

def verify_pipeline():
    print("Starting pipeline verification...")

    config_path = os.path.join(
        PROJECT_DIR, "configs", "config.yaml"
    )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("Configuration loaded successfully!")

    required_modules = [
        "src.ingestion.data_loader",
        "src.preparation.clean_data",
        "src.preparation.split_data",
        "src.transformation.transform_data",
        "src.features.feature_store",
        "src.training.popularity_model",
        "src.training.svd_model",
        "src.evaluation.metrics",
        "src.evaluation.evaluate"
    ]

    for module_name in required_modules:
        __import__(module_name)
        print("Verified:", module_name)

    required_files = [
        "data/raw/ratings.csv",
        "data/raw/movies.csv",
        "data/raw/users.csv",
        "models/popularity_model.pkl",
        "models/svd_model.pkl",
        "reports/model_evaluation_results.csv"
    ]

    for relative_path in required_files:
        file_path = os.path.join(
            PROJECT_DIR, relative_path
        )

        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Missing required file: {relative_path}"
            )

        print("Found:", relative_path)

    print("\nPipeline verification completed successfully!")


if __name__ == "__main__":
    verify_pipeline()
