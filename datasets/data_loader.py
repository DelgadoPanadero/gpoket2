import os
from datasets import load_dataset, IterableDataset

# Define the default local path. This is used when no environment variable is set.
DEFAULT_LOCAL_PATH = "./data/gold_materialized/"

def load_gold_stream(streaming: bool = True) -> IterableDataset:
    """
    Loads the final, model-ready Gold dataset as a stream.

    This function is environment-aware. It will load from the path specified
    in the GOLD_DATASET_PATH environment variable. If the variable is not set,
    it defaults to loading from a local directory.

    This allows the training script to be completely agnostic to the data source.

    Args:
        streaming (bool): Whether to load the data in streaming mode.

    Returns:
        An iterable dataset ready for training.
    """
    # Get the data path from the environment, or use the local default
    data_path = os.getenv("GOLD_DATASET_PATH", DEFAULT_LOCAL_PATH)

    print(f"Loading Gold dataset from source: {data_path}")

    if not os.path.exists(data_path) and not data_path.startswith("s3://"):
        raise FileNotFoundError(
            f"Local data not found at '{data_path}'. "
            "Did you run 'python materialize_gold_locally.py' first?"
        )

    # The format is 'arrow' if loading from a saved directory (local or S3)
    dataset = load_dataset(
        "arrow",
        data_files=f"{data_path}*.arrow",
        streaming=streaming
    )

    return dataset