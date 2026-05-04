import torch
from transformers import DataCollatorForLanguageModeling


class ConditionedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        pokemon_idx = [f.pop("pokemon_idx") for f in features]
        batch = super().__call__(features)
        batch["pokemon_idx"] = torch.tensor(pokemon_idx, dtype=torch.long)
        return batch
