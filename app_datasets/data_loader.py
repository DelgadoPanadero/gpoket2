import pandas as pd
from datasets import load_dataset

def load_pokemons(layer: str = "gold", split: str = "train") -> pd.DataFrame:
    """
    Loads a specific layer of the Pokemon dataset as a pandas DataFrame.

    This is the standard function for accessing project data. It defaults to
    loading the model-ready "gold" layer. The underlying data processing and
    caching for all layers are handled automatically.

    Args:
        layer (str): The data layer to load ('bronze', 'silver', or 'gold').
                     Defaults to 'gold'.
        split (str): The dataset split to load (e.g., 'train').

    Returns:
        A pandas DataFrame representing the requested data layer.
    """
    layer = layer.lower()
    valid_layers = ["bronze", "silver", "gold"]
    if layer not in valid_layers:
        raise ValueError(f"Invalid layer '{layer}'. Choose from {valid_layers}.")

    # The path points to the respective directory containing the loading script.
    # This single line triggers the entire Bronze->Silver->Gold chain if needed.
    dataset_path = f"./datasets/{layer}_pokemon"
    hf_dataset = load_dataset(path=dataset_path, split=split, trust_remote_code=True) # trust_remote_code is needed for local scripts

    # Standardize the output format to a pandas DataFrame
    return hf_dataset.to_pandas()